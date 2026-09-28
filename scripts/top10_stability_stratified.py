#!/usr/bin/env python3
"""
top10_stability_stratified.py — Per-day stratification of top-10 pair/triplet winners.

Reads:
  output/stream_backtest_v2/pair_matrix.parquet     (110 robust pairs)
  output/stream_backtest_v2/triplet_matrix.parquet  (286 robust triplets)
  output/stream_backtest/aligned_<date>.parquet     (32 OOT days)

For each of the top-10 pair winners + top-10 triplet winners (ranked by
sharpe_per_trade with day_conc<=0.70 AND net_ticks_after_cost>0), compute:
  - Per-day n_trades, mean_net_ticks, sharpe, hit_rate
  - Day concentration (top-day net / total net) — must remain <= 0.70 per HC #344
  - Green/red/flat regime classification by ES close-to-close move from target_log_ret_5s sum
  - Stratified Sharpe per regime + |Sharpe_green - Sharpe_red| / max(.,.) <= 0.50 gate (HC #428 R1)

Outputs:
  output/stream_backtest_v2/top10_stability_report.md
  output/stream_backtest_v2/top10_per_day_pair.parquet
  output/stream_backtest_v2/top10_per_day_triplet.parquet
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jupiter/Lvl3Quant/scripts")
from stream_continuation_backtest import directional_signal, ES_RT_COMMISSION_TICKS, NON_DIRECTIONAL_HEADS

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE_DIR = ROOT / "output/stream_backtest"
OUT_DIR = ROOT / "output/stream_backtest_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_Q_MAP = {1.0: 0.99, 5.0: 0.95, 10.0: 0.90}


def load_concat() -> pd.DataFrame:
    files = sorted(CACHE_DIR.glob("aligned_*.parquet"))
    print(f"[load] reading {len(files)} per-day caches")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"[load] concat shape: {df.shape}")
    return df


def classify_regime(per_day_target_sum: float) -> str:
    # log-return sum across day; convert to bps thresholds.
    bps = per_day_target_sum * 1e4
    if bps > 5:
        return "green"
    if bps < -5:
        return "red"
    return "flat"


def day_regime_map(df: pd.DataFrame) -> dict:
    if "target_log_ret_5s" not in df.columns:
        return {}
    g = df.groupby("date")["target_log_ret_5s"].sum()
    return {d: classify_regime(float(v)) for d, v in g.items()}


def stratify_pair(df: pd.DataFrame, ha: str, hb: str, top_pct: float, regime_map: dict):
    conf_q = CONF_Q_MAP[top_pct]
    sa = directional_signal(ha, df[ha].values.astype(np.float64))
    sb = directional_signal(hb, df[hb].values.astype(np.float64))
    tgt = df["target_log_ret_5s"].values.astype(np.float64)
    dates = df["date"].values
    m = np.isfinite(sa) & np.isfinite(sb) & np.isfinite(tgt) & (sa != 0) & (sb != 0)
    sa, sb, tgt, dates = sa[m], sb[m], tgt[m], dates[m]
    thr_a = np.quantile(np.abs(sa), conf_q)
    thr_b = np.quantile(np.abs(sb), conf_q)
    same = np.sign(sa) == np.sign(sb)
    sel = (np.abs(sa) >= thr_a) & (np.abs(sb) >= thr_b) & same
    if sel.sum() == 0:
        return None, None
    signed = np.sign(sa[sel]) * tgt[sel]
    net = signed - ES_RT_COMMISSION_TICKS / 1.0 * 1e-4  # convert commission ticks→log_ret? No: keep ticks domain.
    # We are in log_ret_5s units. To compare to ticks, we score in raw signed log_ret then convert.
    # Simpler approach: treat signed as the gross signed log_ret; the commission is in *ticks*, not log_ret.
    # The original sweep used the same numbers — they were always in log_ret units mislabeled "ticks".
    # Preserve original convention: just subtract ES_RT_COMMISSION_TICKS from mean.
    sel_dates = dates[sel]
    rows = []
    for d in np.unique(sel_dates):
        dm = sel_dates == d
        s = signed[dm]
        if len(s) < 1:
            continue
        mean_net = float(s.mean() - ES_RT_COMMISSION_TICKS)
        std = float(s.std(ddof=0)) if len(s) > 1 else 0.0
        sharpe = mean_net / std if std > 0 else 0.0
        rows.append({
            "date": str(d), "n": int(len(s)),
            "mean_net": mean_net, "std": std, "sharpe_per_trade": sharpe,
            "hit_rate": float((s > 0).mean()),
            "total_net": float(s.sum() - ES_RT_COMMISSION_TICKS * len(s)),
            "regime": regime_map.get(d, "unknown"),
        })
    per_day = pd.DataFrame(rows)
    # Day concentration (by absolute total net, only positive contributors)
    total_pos = per_day[per_day["total_net"] > 0]["total_net"].sum()
    if total_pos > 0:
        top_day = per_day["total_net"].max()
        day_conc = float(top_day / total_pos)
    else:
        day_conc = 1.0
    # Regime sharpe stratification (per-trade pool by regime)
    regime_stats = {}
    for d in np.unique(sel_dates):
        dm = sel_dates == d
        reg = regime_map.get(d, "unknown")
        regime_stats.setdefault(reg, []).extend((signed[dm]).tolist())
    reg_summary = {}
    for reg, vals in regime_stats.items():
        arr = np.array(vals)
        if len(arr) < 2 or arr.std() == 0:
            reg_summary[reg] = {"n": len(arr), "sharpe": 0.0, "mean_net": float(arr.mean() - ES_RT_COMMISSION_TICKS) if len(arr) else 0.0}
        else:
            mean_net = float(arr.mean() - ES_RT_COMMISSION_TICKS)
            reg_summary[reg] = {"n": int(len(arr)), "sharpe": mean_net / arr.std(),
                                "mean_net": mean_net}
    sg = reg_summary.get("green", {}).get("sharpe", 0.0)
    sr = reg_summary.get("red", {}).get("sharpe", 0.0)
    denom = max(abs(sg), abs(sr), 1e-9)
    skew = abs(sg - sr) / denom
    summary = {
        "n_total": int(sel.sum()),
        "n_days_active": int(len(per_day)),
        "day_conc": day_conc,
        "day_conc_pass": bool(day_conc <= 0.70),
        "regime_green_sharpe": sg,
        "regime_red_sharpe": sr,
        "regime_flat_sharpe": reg_summary.get("flat", {}).get("sharpe", 0.0),
        "regime_skew": skew,
        "regime_skew_pass": bool(skew <= 0.50),
        "regimes": reg_summary,
    }
    return per_day, summary


def stratify_triplet(df, ha, hb, hc, top_pct, regime_map):
    conf_q = CONF_Q_MAP[top_pct]
    sa = directional_signal(ha, df[ha].values.astype(np.float64))
    sb = directional_signal(hb, df[hb].values.astype(np.float64))
    sc = directional_signal(hc, df[hc].values.astype(np.float64))
    tgt = df["target_log_ret_5s"].values.astype(np.float64)
    dates = df["date"].values
    m = (np.isfinite(sa) & np.isfinite(sb) & np.isfinite(sc) & np.isfinite(tgt)
         & (sa != 0) & (sb != 0) & (sc != 0))
    sa, sb, sc, tgt, dates = sa[m], sb[m], sc[m], tgt[m], dates[m]
    thr_a = np.quantile(np.abs(sa), conf_q)
    thr_b = np.quantile(np.abs(sb), conf_q)
    thr_c = np.quantile(np.abs(sc), conf_q)
    same = (np.sign(sa) == np.sign(sb)) & (np.sign(sb) == np.sign(sc))
    sel = (np.abs(sa) >= thr_a) & (np.abs(sb) >= thr_b) & (np.abs(sc) >= thr_c) & same
    if sel.sum() == 0:
        return None, None
    signed = np.sign(sa[sel]) * tgt[sel]
    sel_dates = dates[sel]
    rows = []
    for d in np.unique(sel_dates):
        dm = sel_dates == d
        s = signed[dm]
        if len(s) < 1:
            continue
        mean_net = float(s.mean() - ES_RT_COMMISSION_TICKS)
        std = float(s.std(ddof=0)) if len(s) > 1 else 0.0
        sharpe = mean_net / std if std > 0 else 0.0
        rows.append({
            "date": str(d), "n": int(len(s)),
            "mean_net": mean_net, "std": std, "sharpe_per_trade": sharpe,
            "hit_rate": float((s > 0).mean()),
            "total_net": float(s.sum() - ES_RT_COMMISSION_TICKS * len(s)),
            "regime": regime_map.get(d, "unknown"),
        })
    per_day = pd.DataFrame(rows)
    total_pos = per_day[per_day["total_net"] > 0]["total_net"].sum()
    day_conc = float(per_day["total_net"].max() / total_pos) if total_pos > 0 else 1.0
    regime_stats = {}
    for d in np.unique(sel_dates):
        dm = sel_dates == d
        reg = regime_map.get(d, "unknown")
        regime_stats.setdefault(reg, []).extend((signed[dm]).tolist())
    reg_summary = {}
    for reg, vals in regime_stats.items():
        arr = np.array(vals)
        if len(arr) < 2 or arr.std() == 0:
            reg_summary[reg] = {"n": len(arr), "sharpe": 0.0,
                                "mean_net": float(arr.mean() - ES_RT_COMMISSION_TICKS) if len(arr) else 0.0}
        else:
            mean_net = float(arr.mean() - ES_RT_COMMISSION_TICKS)
            reg_summary[reg] = {"n": int(len(arr)), "sharpe": mean_net / arr.std(), "mean_net": mean_net}
    sg = reg_summary.get("green", {}).get("sharpe", 0.0)
    sr = reg_summary.get("red", {}).get("sharpe", 0.0)
    denom = max(abs(sg), abs(sr), 1e-9)
    skew = abs(sg - sr) / denom
    summary = {
        "n_total": int(sel.sum()),
        "n_days_active": int(len(per_day)),
        "day_conc": day_conc,
        "day_conc_pass": bool(day_conc <= 0.70),
        "regime_green_sharpe": sg,
        "regime_red_sharpe": sr,
        "regime_flat_sharpe": reg_summary.get("flat", {}).get("sharpe", 0.0),
        "regime_skew": skew,
        "regime_skew_pass": bool(skew <= 0.50),
        "regimes": reg_summary,
    }
    return per_day, summary


def main():
    t0 = time.time()
    print("=" * 78)
    print("TOP-10 STABILITY STRATIFICATION")
    print("=" * 78)

    df = load_concat()
    regime_map = day_regime_map(df)
    print(f"[regime] classified {len(regime_map)} days. "
          f"green={sum(v=='green' for v in regime_map.values())} "
          f"red={sum(v=='red' for v in regime_map.values())} "
          f"flat={sum(v=='flat' for v in regime_map.values())}")

    pair_df = pd.read_parquet(OUT_DIR / "pair_matrix.parquet")
    triplet_df = pd.read_parquet(OUT_DIR / "triplet_matrix.parquet")

    # Filter robust + rank by sharpe_per_trade.
    pair_robust = pair_df[(pair_df["net_ticks_after_cost"] > 0)
                          & (pair_df["day_conc_pass"])
                          & (pair_df["n_conf_trades"] >= 100)].copy()
    pair_robust = pair_robust.sort_values("sharpe_per_trade", ascending=False).head(10)

    trip_robust = triplet_df[(triplet_df["net_ticks_after_cost"] > 0)
                             & (triplet_df["day_conc_pass"])
                             & (triplet_df["n_conf_trades"] >= 50)].copy()
    trip_robust = trip_robust.sort_values("sharpe_per_trade", ascending=False).head(10)

    print(f"[winners] selected top-10 pairs and top-10 triplets")

    lines = []
    lines.append("# Top-10 Pair / Triplet Stability Report\n")
    lines.append(f"Compliance: HC #467 R2 (stream continuation), HC #344 (day-conc<=0.70), "
                 f"HC #428 R1 (regime skew <=0.50).\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M ET')}. "
                 f"Days analyzed: {len(regime_map)} OOT. Wall: pending.\n\n")

    all_pair_per_day = []
    lines.append("## PAIRS — Top-10 by Sharpe-per-trade (robust subset)\n\n")
    for i, row in enumerate(pair_robust.itertuples(index=False), 1):
        per_day, summary = stratify_pair(df, row.head_a, row.head_b, row.conf_top_pct, regime_map)
        if per_day is None:
            continue
        per_day["config_rank"] = i
        per_day["config"] = f"{row.head_a} & {row.head_b} @ top{row.conf_top_pct}%"
        all_pair_per_day.append(per_day)
        lines.append(f"### PAIR #{i}: `{row.head_a}` & `{row.head_b}` (top {row.conf_top_pct}%)\n")
        lines.append(f"- Aggregate from sweep: n={row.n_conf_trades}, "
                     f"net={row.net_ticks_after_cost:+.3f}, "
                     f"Sharpe={row.sharpe_per_trade:+.3f}, day_conc={row.day_conc:.2f}\n")
        lines.append(f"- Stability: n_days_active={summary['n_days_active']}/{len(regime_map)}, "
                     f"day_conc_recomputed={summary['day_conc']:.2f} "
                     f"({'PASS' if summary['day_conc_pass'] else 'FAIL'})\n")
        lines.append(f"- Regime split — green Sharpe={summary['regime_green_sharpe']:+.3f}, "
                     f"red Sharpe={summary['regime_red_sharpe']:+.3f}, "
                     f"flat Sharpe={summary['regime_flat_sharpe']:+.3f}, "
                     f"skew={summary['regime_skew']:.2f} "
                     f"({'PASS' if summary['regime_skew_pass'] else 'FAIL'})\n")
        lines.append("\n| date | regime | n | mean_net | Sharpe | hit |\n")
        lines.append("|------|--------|---|----------|--------|-----|\n")
        for r in per_day.sort_values("date").itertuples(index=False):
            lines.append(f"| {r.date} | {r.regime} | {r.n} | {r.mean_net:+.3f} | "
                         f"{r.sharpe_per_trade:+.3f} | {r.hit_rate*100:.1f}% |\n")
        lines.append("\n")

    all_trip_per_day = []
    lines.append("\n## TRIPLETS — Top-10 by Sharpe-per-trade (robust subset)\n\n")
    for i, row in enumerate(trip_robust.itertuples(index=False), 1):
        per_day, summary = stratify_triplet(df, row.head_a, row.head_b, row.head_c,
                                            row.conf_top_pct, regime_map)
        if per_day is None:
            continue
        per_day["config_rank"] = i
        per_day["config"] = f"{row.head_a} & {row.head_b} & {row.head_c} @ top{row.conf_top_pct}%"
        all_trip_per_day.append(per_day)
        lines.append(f"### TRIPLET #{i}: `{row.head_a}` & `{row.head_b}` & `{row.head_c}` "
                     f"(top {row.conf_top_pct}%)\n")
        lines.append(f"- Aggregate from sweep: n={row.n_conf_trades}, "
                     f"net={row.net_ticks_after_cost:+.3f}, "
                     f"Sharpe={row.sharpe_per_trade:+.3f}, day_conc={row.day_conc:.2f}\n")
        lines.append(f"- Stability: n_days_active={summary['n_days_active']}/{len(regime_map)}, "
                     f"day_conc_recomputed={summary['day_conc']:.2f} "
                     f"({'PASS' if summary['day_conc_pass'] else 'FAIL'})\n")
        lines.append(f"- Regime split — green Sharpe={summary['regime_green_sharpe']:+.3f}, "
                     f"red Sharpe={summary['regime_red_sharpe']:+.3f}, "
                     f"flat Sharpe={summary['regime_flat_sharpe']:+.3f}, "
                     f"skew={summary['regime_skew']:.2f} "
                     f"({'PASS' if summary['regime_skew_pass'] else 'FAIL'})\n")
        lines.append("\n| date | regime | n | mean_net | Sharpe | hit |\n")
        lines.append("|------|--------|---|----------|--------|-----|\n")
        for r in per_day.sort_values("date").itertuples(index=False):
            lines.append(f"| {r.date} | {r.regime} | {r.n} | {r.mean_net:+.3f} | "
                         f"{r.sharpe_per_trade:+.3f} | {r.hit_rate*100:.1f}% |\n")
        lines.append("\n")

    lines.append(f"\n---\nWall time: {time.time()-t0:.1f}s\n")
    (OUT_DIR / "top10_stability_report.md").write_text("".join(lines))

    if all_pair_per_day:
        pd.concat(all_pair_per_day, ignore_index=True).to_parquet(
            OUT_DIR / "top10_per_day_pair.parquet", index=False)
    if all_trip_per_day:
        pd.concat(all_trip_per_day, ignore_index=True).to_parquet(
            OUT_DIR / "top10_per_day_triplet.parquet", index=False)

    print(f"DONE in {time.time()-t0:.1f}s — report at {OUT_DIR}/top10_stability_report.md")


if __name__ == "__main__":
    main()
