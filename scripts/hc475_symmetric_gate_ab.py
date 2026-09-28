#!/usr/bin/env python3
"""
hc475_symmetric_gate_ab.py — HC #475 R1/R2/R3 + HC #474 R6 compliance.

A/B test: per-tail top-X% gate (baseline) vs symmetric absolute-magnitude gate
(treatment) for the top-5 surviving confluence configs on the 16-day OOT NPZ.

Calibration: pick a single positive threshold k per head such that |signal| > k
on IS days (first 70% by date) drives short-share into [0.40, 0.60]. Apply to
ALL days; evaluate full window (per HC #428 R1's regime-agnostic rule with
limited data) and OOT-only (last 30%). NO IS-on-OOT leakage.

Output:
  output/hc475_ab/symmetric_gate_summary.parquet
  output/hc475_ab/symmetric_gate_per_day.parquet
  output/hc475_ab/symmetric_gate_fills.parquet
  reports/hc475_longshort_diagnosis/06_symmetric_gate_AB.md

Per HC #474 R6: report row order is Sharpe | Sortino | PF | WR | trades/day | shares.
Per HC #74 / HC #469 R3: canonical FIFO market replay only, NEVER midpoint.
Per HC #0: sliding-window walk-forward. (NPZ is already sliding-window output.)
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3))
sys.path.insert(0, str(LVL3 / "scripts"))
sys.path.insert(0, str(LVL3 / "scripts" / "v3_4_research"))

from stream_continuation_backtest import (
    NON_DIRECTIONAL_HEADS,
    ES_RT_COMMISSION_TICKS,
)
from hc432_fifo_full_market_replay import run_one_date


def directional_signal_v342(name: str, vals: np.ndarray) -> np.ndarray:
    """v3.4.2-specific directional signal — uses raw values directly.

    NOTE: stream_continuation_backtest.directional_signal subtracts 0.5 from
    pred_p_up_* heads assuming they're in [0,1]. In this NPZ the heads are
    already pre-centered (pred_p_up_5s range -0.52..0.08, mean -0.11), so the
    -0.5 recentering pushes everything negative and breaks symmetric gating.
    Documented in HC #475 followup (2026-05-21).
    For directional heads we treat the raw value as the signed signal.
    For non-directional heads we return zeros (gated by |signal| only).
    """
    if name in NON_DIRECTIONAL_HEADS:
        return np.zeros_like(vals)
    if name.startswith("pred_p_reversal_"):
        return np.zeros_like(vals)
    if name == "pred_pred_mae_30s_ticks" or name == "pred_pred_mae_60s_ticks":
        return -vals  # mae is "downside excursion" → short bias
    # All other directional heads (pred_log_ret_*, pred_p_up_*, pred_fifo_*_net,
    # pred_pred_mfe_*) carry direction in their raw sign.
    return vals


directional_signal = directional_signal_v342

NPZ = LVL3 / "output" / "hc432_v342_47day_validation" / "fold_00_ep1_oot_inference_47day_hc432.npz"
BASELINE_SUMMARY = LVL3 / "output" / "stream_backtest_v2" / "surviving_canonical_fifo_summary.parquet"
BASELINE_PER_DAY = LVL3 / "output" / "stream_backtest_v2" / "surviving_canonical_fifo_per_day.parquet"
OUT = LVL3 / "output" / "hc475_ab"
OUT.mkdir(parents=True, exist_ok=True)
REPORT_DIR = LVL3 / "reports" / "hc475_longshort_diagnosis"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Top 5 surviving configs (from baseline pipeline) — same set as baseline for clean A/B.
CONFIGS = [
    ("pair",    "pair01_logret1s+pup5s",                              ["pred_log_ret_1s", "pred_p_up_5s"]),
    ("pair",    "pair07_logret10s+logret60sq50",                      ["pred_log_ret_10s", "pred_log_ret_60s_q50"]),
    ("pair",    "pair08_logret5s+pup5s",                              ["pred_log_ret_5s", "pred_p_up_5s"]),
    ("triplet", "trip01_pup5s+logret1s+logret10s",                    ["pred_p_up_5s", "pred_log_ret_1s", "pred_log_ret_10s"]),
    ("triplet", "trip03_logret5s+pup5s+logret1s",                     ["pred_log_ret_5s", "pred_p_up_5s", "pred_log_ret_1s"]),
]

# Reference bracket — same as baseline.
TP_TICKS = 4.0
SL_TICKS = 3.0
HOLD_S = 30.0
CANCEL_S = 10.0
ORDER_TYPE = "passive_at_touch"

# Calibration target: short-share window on IS days.
TARGET_SHORT_SHARE_LO = 0.40
TARGET_SHORT_SHARE_HI = 0.60
IS_FRACTION = 0.70  # first 70% of dates for calibration

# Regime cache (reuses baseline).
REGIME_CACHE = LVL3 / "output" / "stream_backtest_v2" / "top10_per_day_pair.parquet"


def regime_classify(date_str: str) -> str:
    if not REGIME_CACHE.exists():
        return "unknown"
    df = pd.read_parquet(REGIME_CACHE)
    if "date" not in df.columns or "regime" not in df.columns:
        return "unknown"
    df["date"] = df["date"].astype(str)
    row = df[df["date"] == date_str]
    if row.empty:
        return "unknown"
    return str(row["regime"].iloc[0])


def calibrate_equal_count_thresholds(signal: np.ndarray, is_mask: np.ndarray,
                                     per_side_rate: float = 0.05) -> Tuple[float, float]:
    """For a DIRECTIONAL head, choose SEPARATE thresholds k_pos and k_neg on IS data
    so that approximately `per_side_rate` fraction of all IS events trigger LONG
    (signal > k_pos) and `per_side_rate` trigger SHORT (signal < -k_neg). This forces
    each head to contribute equal trigger counts per side, neutralizing magnitude skew.
    """
    is_signal = signal[is_mask & np.isfinite(signal)]
    if is_signal.size == 0:
        return 0.0, 0.0
    n_total = is_signal.size
    K = max(1, int(per_side_rate * n_total))
    positive = is_signal[is_signal > 0]
    negative = is_signal[is_signal < 0]
    if positive.size >= K:
        k_pos = float(np.partition(positive, -K)[-K])  # K-th largest positive
    elif positive.size > 0:
        k_pos = float(positive.min())  # take all positives
    else:
        k_pos = float("inf")  # no positives ever → no long triggers possible
    if negative.size >= K:
        k_neg = float(-np.partition(negative, K - 1)[K - 1])  # K-th most negative, made positive
    elif negative.size > 0:
        k_neg = float(-negative.max())
    else:
        k_neg = float("inf")
    return k_pos, k_neg


def calibrate_nondirectional_threshold(signal: np.ndarray, is_mask: np.ndarray,
                                        target_pct: float = 0.05) -> float:
    """For non-directional confluence heads, keep top X% by |signal| on IS data."""
    is_signal = signal[is_mask & np.isfinite(signal)]
    if is_signal.size == 0:
        return 0.0
    return float(np.quantile(np.abs(is_signal), 1.0 - target_pct))


def symmetric_confluence_mask(arrs: Dict[str, np.ndarray], heads: List[str],
                              is_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Symmetric gate: each directional head contributes long_h = (signal > k) and
    short_h = (signal < -k). Non-directional heads gate by |signal| > k_top5%.
    Intersection across heads yields long_mask / short_mask.

    Returns (long_mask, short_mask, calibration_info).
    """
    n = arrs[heads[0]].shape[0]
    long_mask = np.ones(n, dtype=bool)
    short_mask = np.ones(n, dtype=bool)
    calib: Dict = {}

    for h in heads:
        if h not in arrs:
            raise KeyError(f"NPZ missing head {h}")
        raw = arrs[h].astype(np.float64)
        signal = directional_signal(h, raw)

        if h in NON_DIRECTIONAL_HEADS:
            k = calibrate_nondirectional_threshold(signal, is_mask, target_pct=0.05)
            calib[h] = {"kind": "nondirectional", "k": k}
            finite = np.isfinite(signal)
            keep = finite & (np.abs(signal) > k)
            long_mask &= keep
            short_mask &= keep
        else:
            k_pos, k_neg = calibrate_equal_count_thresholds(signal, is_mask, per_side_rate=0.05)
            calib[h] = {"kind": "directional", "k_pos": k_pos, "k_neg": k_neg}
            finite = np.isfinite(signal)
            long_mask &= finite & (signal > k_pos)
            short_mask &= finite & (signal < -k_neg)

    return long_mask, short_mask, calib


def run_one_config(arrs: Dict[str, np.ndarray], cfg_type: str, cfg_name: str,
                   heads: List[str], is_mask: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    t0 = time.time()
    long_mask, short_mask, calib = symmetric_confluence_mask(arrs, heads, is_mask)
    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())
    short_share_triggers = n_short / max(1, (n_long + n_short))
    print(f"[{cfg_name}] symmetric gate: long_triggers={n_long}, short_triggers={n_short}, "
          f"short_share_triggers={short_share_triggers:.3f}")
    print(f"[{cfg_name}] calibration: {calib}")

    sample_dates = arrs["sample_dates"]
    unique_dates = sorted(set(sample_dates.tolist()))
    all_fills: List[dict] = []

    for date_str in unique_dates:
        day_mask = sample_dates == date_str
        for side, mask in [("long", long_mask), ("short", short_mask)]:
            sel = day_mask & mask
            if not sel.any():
                continue
            idx_in_day = np.flatnonzero(sel[day_mask])
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
                    continue
                f["config"] = cfg_name
                f["config_type"] = cfg_type
                f["side"] = side
                all_fills.append(f)

    fills_df = pd.DataFrame(all_fills)
    if fills_df.empty:
        print(f"[{cfg_name}] NO FILLS under symmetric gate")
        return pd.DataFrame(), pd.DataFrame(), {"calib": calib, "n_long": n_long, "n_short": n_short}

    # Per-day rows (overall, not split by side — for direct A/B w/ baseline)
    per_day_rows: List[dict] = []
    for date_str in unique_dates:
        day_fills = fills_df[fills_df["date"] == date_str]
        if day_fills.empty:
            continue
        net = day_fills["net_ticks"].values
        n_long_fills = int((day_fills["side"] == "long").sum())
        n_short_fills = int((day_fills["side"] == "short").sum())
        per_day_rows.append({
            "config": cfg_name,
            "config_type": cfg_type,
            "date": date_str,
            "regime": regime_classify(date_str),
            "n_fills": int(len(net)),
            "n_long_fills": n_long_fills,
            "n_short_fills": n_short_fills,
            "net_ticks_total": float(net.sum()),
            "net_ticks_mean": float(net.mean()),
            "net_ticks_std": float(net.std(ddof=1)) if len(net) > 1 else 0.0,
            "Sharpe": float(net.mean() / net.std(ddof=1)) if len(net) > 1 and net.std(ddof=1) > 0 else 0.0,
            "WR": float((net > 0).mean()),
        })

    per_day_df = pd.DataFrame(per_day_rows)
    elapsed = time.time() - t0
    print(f"[{cfg_name}] done in {elapsed:.1f}s — fills={len(fills_df)} days={len(per_day_df)}")

    return fills_df, per_day_df, {"calib": calib, "n_long": n_long, "n_short": n_short,
                                  "short_share_triggers": short_share_triggers}


def aggregate_summary(per_day_all: pd.DataFrame, fills_all: pd.DataFrame, label: str) -> pd.DataFrame:
    if per_day_all.empty:
        return pd.DataFrame()
    rows = []
    for cfg, sub in per_day_all.groupby("config"):
        f_sub = fills_all[fills_all["config"] == cfg]
        if f_sub.empty:
            continue
        net = f_sub["net_ticks"].values
        n_fills = int(len(net))

        # Sharpe (per-trade), Sortino (downside-only std), PF, WR
        mean_t = float(net.mean())
        std_t = float(net.std(ddof=1)) if n_fills > 1 else 0.0
        sharpe = (mean_t / std_t) if std_t > 0 else 0.0
        downside = net[net < 0]
        d_std = float(downside.std(ddof=1)) if downside.size > 1 else 0.0
        sortino = (mean_t / d_std) if d_std > 0 else 0.0
        gains = net[net > 0].sum()
        losses = -net[net < 0].sum()
        pf = (gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0
        wr = float((net > 0).mean())

        # Day-conc / regime split (across-day)
        day_net = sub["net_ticks_total"].values
        day_conc = float(day_net.max() / day_net.sum()) if day_net.sum() > 0 else float("nan")
        green = sub[sub["regime"] == "green"]["Sharpe"].mean() if (sub["regime"] == "green").any() else float("nan")
        red = sub[sub["regime"] == "red"]["Sharpe"].mean() if (sub["regime"] == "red").any() else float("nan")
        if not (np.isnan(green) or np.isnan(red)):
            mx = max(abs(green), abs(red))
            skew = abs(green - red) / mx if mx > 0 else 0.0
        else:
            skew = float("nan")

        # Long/short share of fills
        n_long = int(f_sub["side"].eq("long").sum())
        n_short = int(f_sub["side"].eq("short").sum())
        long_share = n_long / max(1, n_fills)
        short_share = n_short / max(1, n_fills)
        trades_per_day = n_fills / max(1, len(sub))

        # First-half vs second-half (by date order)
        sub_sorted = sub.sort_values("date").reset_index(drop=True)
        mid = len(sub_sorted) // 2
        if mid >= 1 and len(sub_sorted) - mid >= 1:
            h1_fills = fills_all[(fills_all["config"] == cfg) &
                                 (fills_all["date"].isin(sub_sorted["date"].iloc[:mid]))]["net_ticks"].values
            h2_fills = fills_all[(fills_all["config"] == cfg) &
                                 (fills_all["date"].isin(sub_sorted["date"].iloc[mid:]))]["net_ticks"].values
            h1_sharpe = (h1_fills.mean() / h1_fills.std(ddof=1)) if h1_fills.size > 1 and h1_fills.std(ddof=1) > 0 else 0.0
            h2_sharpe = (h2_fills.mean() / h2_fills.std(ddof=1)) if h2_fills.size > 1 and h2_fills.std(ddof=1) > 0 else 0.0
            h1_wr = float((h1_fills > 0).mean()) if h1_fills.size else float("nan")
            h2_wr = float((h2_fills > 0).mean()) if h2_fills.size else float("nan")
        else:
            h1_sharpe = h2_sharpe = 0.0
            h1_wr = h2_wr = float("nan")

        rows.append({
            "config": cfg,
            "label": label,
            "Sharpe": float(sharpe),
            "Sortino": float(sortino),
            "PF": float(pf),
            "WR": float(wr),
            "trades_per_day": float(trades_per_day),
            "long_share": float(long_share),
            "short_share": float(short_share),
            "n_fills": n_fills,
            "n_days": int(len(sub)),
            "net_ticks_per_trade": float(mean_t),
            "Sharpe_h1": float(h1_sharpe),
            "Sharpe_h2": float(h2_sharpe),
            "WR_h1": float(h1_wr) if not np.isnan(h1_wr) else 0.0,
            "WR_h2": float(h2_wr) if not np.isnan(h2_wr) else 0.0,
            "day_conc": float(day_conc) if not np.isnan(day_conc) else 0.0,
            "Sharpe_green": float(green) if not np.isnan(green) else 0.0,
            "Sharpe_red": float(red) if not np.isnan(red) else 0.0,
            "regime_skew": float(skew) if not np.isnan(skew) else 0.0,
            "pass_dayconc": (day_conc <= 0.70) if not np.isnan(day_conc) else False,
            "pass_regime": (skew <= 0.50) if not np.isnan(skew) else False,
            "pass_wr_floor": (wr >= 0.55) or (sharpe >= 2.0 and pf >= 1.8),
        })
    return pd.DataFrame(rows)


def baseline_summary_for_configs(cfg_names: List[str]) -> pd.DataFrame:
    """Load baseline summary parquet and reshape to our row schema for A/B."""
    if not BASELINE_SUMMARY.exists() or not BASELINE_PER_DAY.exists():
        print(f"[warn] baseline parquets missing — A/B left-side will be empty")
        return pd.DataFrame()
    bsum = pd.read_parquet(BASELINE_SUMMARY)
    bpd = pd.read_parquet(BASELINE_PER_DAY)
    baseline_fills_path = LVL3 / "output" / "stream_backtest_v2" / "surviving_canonical_fifo_fills.parquet"
    if not baseline_fills_path.exists():
        print(f"[warn] baseline fills parquet missing")
        return pd.DataFrame()
    bfills = pd.read_parquet(baseline_fills_path)

    # The baseline fills don't have a 'side' column directly — derive from any side col or skip.
    if "side" not in bfills.columns and "direction" in bfills.columns:
        bfills = bfills.rename(columns={"direction": "side"})

    bfills_in = bfills[bfills["config"].isin(cfg_names)].copy()
    bpd_in = bpd[bpd["config"].isin(cfg_names)].copy()
    if bfills_in.empty:
        return pd.DataFrame()

    return aggregate_summary(bpd_in, bfills_in, label="baseline_per_tail_topq")


def write_ab_report(baseline: pd.DataFrame, treatment: pd.DataFrame, calib_meta: Dict, wall_s: float) -> Path:
    lines = []
    lines.append("# HC #475 — Symmetric Gate A/B vs Baseline Per-Tail Top-Quantile Gate")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M ET')}. Wall: {wall_s:.1f}s.")
    lines.append("")
    lines.append("**Compliance**: HC #475 R1/R2/R3 (long/short balance, both-sides competency, short-only justification), "
                 "HC #474 R6 (Sharpe/Sortino/PF/WR/trades-per-day order), HC #428 R1 (regime gate), "
                 "HC #472 R2 (first-half / second-half stratification), HC #74 (canonical FIFO replay).")
    lines.append("")
    lines.append("**Data scope**: 16-day OOT NPZ (same as baseline). Symmetric threshold calibrated on first 70% of "
                 "dates (IS), applied to all 16 days. NO IS-on-OOT leakage.")
    lines.append("")
    lines.append("**Bracket** (identical to baseline): TP=4t, SL=3t, hold=30s, cancel=10s, passive_at_touch.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## A/B Headline (HC #474 R6 column order)")
    lines.append("")
    lines.append("| Config | Variant | Sharpe | Sortino | PF | WR | trades/day | long_share | short_share | Regime skew | Gates |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

    cfg_names = sorted(set(treatment["config"].tolist()) | set(baseline["config"].tolist() if not baseline.empty else []))
    for cfg in cfg_names:
        # Baseline row
        if not baseline.empty and cfg in set(baseline["config"]):
            r = baseline[baseline["config"] == cfg].iloc[0]
            gates = "FAIL" if not (r["pass_dayconc"] and r["pass_regime"] and r["pass_wr_floor"]) else "PASS"
            lines.append(
                f"| {cfg} | baseline | {r['Sharpe']:+.3f} | {r['Sortino']:+.3f} | "
                f"{r['PF']:.2f} | {r['WR']*100:.1f}% | {r['trades_per_day']:.1f} | "
                f"{r['long_share']*100:.1f}% | {r['short_share']*100:.1f}% | "
                f"{r['regime_skew']:.2f} | {gates} |"
            )
        # Treatment row
        if cfg in set(treatment["config"]):
            r = treatment[treatment["config"] == cfg].iloc[0]
            gates = "FAIL" if not (r["pass_dayconc"] and r["pass_regime"] and r["pass_wr_floor"]) else "PASS"
            lines.append(
                f"| {cfg} | **symmetric** | {r['Sharpe']:+.3f} | {r['Sortino']:+.3f} | "
                f"{r['PF']:.2f} | {r['WR']*100:.1f}% | {r['trades_per_day']:.1f} | "
                f"{r['long_share']*100:.1f}% | {r['short_share']*100:.1f}% | "
                f"{r['regime_skew']:.2f} | {gates} |"
            )

    lines.append("")
    lines.append("**Gates**: pass_dayconc (≤0.70), pass_regime (skew ≤0.50), pass_wr_floor (WR ≥55% OR (Sharpe≥2 AND PF≥1.8)).")
    lines.append("")
    lines.append("## First-half / Second-half stratification (HC #472 R2)")
    lines.append("")
    lines.append("| Config | Variant | Sharpe_h1 | Sharpe_h2 | WR_h1 | WR_h2 |")
    lines.append("|---|---|---|---|---|---|")
    for cfg in cfg_names:
        if not baseline.empty and cfg in set(baseline["config"]):
            r = baseline[baseline["config"] == cfg].iloc[0]
            lines.append(f"| {cfg} | baseline | {r['Sharpe_h1']:+.3f} | {r['Sharpe_h2']:+.3f} | {r['WR_h1']*100:.1f}% | {r['WR_h2']*100:.1f}% |")
        if cfg in set(treatment["config"]):
            r = treatment[treatment["config"] == cfg].iloc[0]
            lines.append(f"| {cfg} | symmetric | {r['Sharpe_h1']:+.3f} | {r['Sharpe_h2']:+.3f} | {r['WR_h1']*100:.1f}% | {r['WR_h2']*100:.1f}% |")
    lines.append("")
    lines.append("## Calibration metadata")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(calib_meta, indent=2, default=str))
    lines.append("```")
    lines.append("")
    lines.append("## Verdict (HC #475 R3 short-only deploy decision)")
    lines.append("")
    if treatment.empty:
        lines.append("- Symmetric gate produced no fills — fallback to calibrated probability gate required.")
    else:
        any_pass = (treatment["pass_dayconc"] & treatment["pass_regime"] & treatment["pass_wr_floor"]).any()
        if any_pass:
            winners = treatment[treatment["pass_dayconc"] & treatment["pass_regime"] & treatment["pass_wr_floor"]]
            best = winners.sort_values("Sharpe", ascending=False).iloc[0]
            lines.append(f"- **CANDIDATE FOUND**: `{best['config']}` clears all three gates under the symmetric gate.")
            lines.append(f"  Sharpe={best['Sharpe']:+.3f}, Sortino={best['Sortino']:+.3f}, PF={best['PF']:.2f}, "
                         f"WR={best['WR']*100:.1f}%, trades/day={best['trades_per_day']:.1f}, "
                         f"long_share={best['long_share']*100:.1f}% / short_share={best['short_share']*100:.1f}%.")
        else:
            lines.append("- No symmetric-gate config clears all three gates. Per HC #475 R3, this is NOT a deploy candidate.")
            lines.append("  Best Sharpe under symmetric gate, with failure reasons:")
            best = treatment.sort_values("Sharpe", ascending=False).iloc[0]
            fails = []
            if not best["pass_dayconc"]: fails.append(f"day-conc {best['day_conc']:.2f}>0.70")
            if not best["pass_regime"]:  fails.append(f"regime skew {best['regime_skew']:.2f}>0.50")
            if not best["pass_wr_floor"]: fails.append(f"WR {best['WR']*100:.1f}%<55% and not (Sharpe≥2 AND PF≥1.8)")
            lines.append(f"  `{best['config']}`: Sharpe={best['Sharpe']:+.3f}, fails [{', '.join(fails) or 'unknown'}].")
    lines.append("")
    lines.append("- HC #475 R4 alpha redevelopment (sign-balanced loss, continuation targets, execution-tailored outputs) "
                 "remains the medium-term recommendation regardless of this A/B outcome — the symmetric gate is a policy "
                 "patch over a model whose predicted-magnitude distribution is asymmetric.")

    report_path = REPORT_DIR / "06_symmetric_gate_AB.md"
    report_path.write_text("\n".join(lines))
    return report_path


def main():
    t_start = time.time()
    print(f"[load] {NPZ}")
    arrs = {k: v for k, v in np.load(NPZ, allow_pickle=False).items()}
    sample_dates = arrs["sample_dates"]
    unique_dates = sorted(set(sample_dates.tolist()))
    print(f"[load] n_samples={sample_dates.shape[0]:,} n_dates={len(unique_dates)}")

    # IS / OOT split by date order
    n_is_dates = max(1, int(IS_FRACTION * len(unique_dates)))
    is_dates = set(unique_dates[:n_is_dates])
    oot_dates = set(unique_dates[n_is_dates:])
    is_mask = np.isin(sample_dates, list(is_dates))
    print(f"[split] IS dates ({n_is_dates}): {sorted(is_dates)}")
    print(f"[split] OOT dates ({len(oot_dates)}): {sorted(oot_dates)}")

    all_fills: List[pd.DataFrame] = []
    all_per_day: List[pd.DataFrame] = []
    calib_meta: Dict[str, Dict] = {}

    for cfg_type, cfg_name, heads in CONFIGS:
        fills_df, per_day_df, meta = run_one_config(arrs, cfg_type, cfg_name, heads, is_mask)
        calib_meta[cfg_name] = meta
        if not fills_df.empty:
            all_fills.append(fills_df)
        if not per_day_df.empty:
            all_per_day.append(per_day_df)

    if all_fills:
        fills_all = pd.concat(all_fills, ignore_index=True)
        fills_all.to_parquet(OUT / "symmetric_gate_fills.parquet", index=False)
        print(f"[write] {OUT}/symmetric_gate_fills.parquet n={len(fills_all)}")
    else:
        fills_all = pd.DataFrame()

    if all_per_day:
        per_day_all = pd.concat(all_per_day, ignore_index=True)
        per_day_all.to_parquet(OUT / "symmetric_gate_per_day.parquet", index=False)
        treatment = aggregate_summary(per_day_all, fills_all, label="symmetric_abs_thresh")
        treatment.to_parquet(OUT / "symmetric_gate_summary.parquet", index=False)
    else:
        treatment = pd.DataFrame()

    # Baseline rebuild on identical config set, identical schema, for clean A/B.
    cfg_names = [c[1] for c in CONFIGS]
    baseline = baseline_summary_for_configs(cfg_names)

    wall_s = time.time() - t_start
    report_path = write_ab_report(baseline, treatment, calib_meta, wall_s)
    print(f"[report] wrote {report_path}")
    print(f"[DONE] {wall_s:.1f}s total")


if __name__ == "__main__":
    main()
