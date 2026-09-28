#!/usr/bin/env python3
"""
HC #271(A) — Re-slice the tp8sl5 winner FIFO result by regime label.

Reads the per-fold table from logs/fifo_replay_v3_top0_001_short_tp8sl5.log
and joins to output/regime_labels/oot_dates_regime.parquet.

Emits a Discord-ready report:
  - Per-fold: date, trend_label, vol_bucket, signals, trades, fill%, WR,
              gross/trade, net/trade, sum_NET_ticks
  - Per-regime aggregates: trend × vol_bucket → sum_NET, n_trades, % folds positive
  - HC #271(A) deployment-gate verdicts:
      a) profitable in ≥1 of {up, down, flat}?
      b) max single-regime concentration ≤50%?
      c) ≥60% per-date positivity?
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
import pandas as pd

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
DEFAULT_LOG = LVL3_ROOT / "logs" / "fifo_replay_v3_top0_001_short_tp8sl5.log"
REGIME_PATH = LVL3_ROOT / "output" / "regime_labels" / "oot_dates_regime.parquet"
DEFAULT_OUT = LVL3_ROOT / "output" / "regime_labels" / "tp8sl5_by_regime.csv"

# Per-fold row format from the FIFO replay log:
#    0   20260406     9     4  44.4% 0.250  -0.500  -0.876  -2.876
# i.e.   fold date    sig   trd  fill%  WR    g/t     n/t     med
ROW_RE = re.compile(
    r"^\s*(?P<fold>\d+)\s+(?P<date>\d{8})\s+(?P<sig>\d+)\s+(?P<trd>\d+)"
    r"(\s+(?P<fillpct>[\d.]+)%\s+(?P<wr>[\d.]+)\s+(?P<gt>[+\-][\d.]+)"
    r"\s+(?P<nt>[+\-][\d.]+)\s+(?P<med>[+\-][\d.]+))?\s*$"
)


def parse_log(path: Path) -> pd.DataFrame:
    rows = []
    if not path.exists():
        print(f"ERROR: log not found {path}", file=sys.stderr)
        sys.exit(1)
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        d = m.groupdict()
        rows.append({
            "fold": int(d["fold"]),
            "date": d["date"],
            "n_signals": int(d["sig"]),
            "n_trades": int(d["trd"]),
            "fill_pct": float(d["fillpct"]) if d["fillpct"] else None,
            "win_rate": float(d["wr"]) if d["wr"] else None,
            "gross_per_trade": float(d["gt"]) if d["gt"] else None,
            "net_per_trade": float(d["nt"]) if d["nt"] else None,
            "median_per_trade": float(d["med"]) if d["med"] else None,
        })
    df = pd.DataFrame(rows)
    df["sum_net_ticks"] = (df["net_per_trade"].fillna(0) * df["n_trades"]).round(2)
    df["sum_gross_ticks"] = (df["gross_per_trade"].fillna(0) * df["n_trades"]).round(2)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default=str(DEFAULT_LOG))
    ap.add_argument("--out-csv", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--label", type=str, default="tp8sl5 short top0.1% passive limit cancel=2s hold=30s",
                    help="Header label for the report")
    args = ap.parse_args()
    LOG_PATH = Path(args.log)
    OUT_PATH = Path(args.out_csv)
    if not REGIME_PATH.exists():
        print(f"WARN: {REGIME_PATH} doesn't exist yet — run regime_label_oot_dates.py first")
        return 1
    folds = parse_log(LOG_PATH)
    regimes = pd.read_parquet(REGIME_PATH)
    merged = folds.merge(regimes, on="date", how="left")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)

    print("=" * 95)
    print(f"PER-FOLD WITH REGIME LABELS — {args.label}")
    print("=" * 95)
    cols = ["fold", "date", "trend_label", "vol_bucket", "close_minus_open_ticks",
            "n_trades", "win_rate", "net_per_trade", "sum_net_ticks"]
    cols = [c for c in cols if c in merged.columns]
    print(merged[cols].to_string(index=False))
    print()

    # Aggregate by trend
    print("=" * 50)
    print("AGGREGATE BY TREND (trend label)")
    print("=" * 50)
    by_trend = merged.groupby("trend_label", dropna=False).agg(
        n_dates=("date", "count"),
        n_trades=("n_trades", "sum"),
        sum_net_ticks=("sum_net_ticks", "sum"),
        sum_gross_ticks=("sum_gross_ticks", "sum"),
        n_dates_positive=("sum_net_ticks", lambda x: int((x > 0).sum())),
    )
    by_trend["pct_dates_positive"] = (
        100.0 * by_trend["n_dates_positive"] / by_trend["n_dates"]
    ).round(1)
    by_trend["sum_net_ticks"] = by_trend["sum_net_ticks"].round(2)
    by_trend["sum_gross_ticks"] = by_trend["sum_gross_ticks"].round(2)
    print(by_trend.to_string())
    print()

    # Aggregate by vol_bucket
    print("=" * 50)
    print("AGGREGATE BY VOL BUCKET")
    print("=" * 50)
    by_vol = merged.groupby("vol_bucket", dropna=False).agg(
        n_dates=("date", "count"),
        n_trades=("n_trades", "sum"),
        sum_net_ticks=("sum_net_ticks", "sum"),
        n_dates_positive=("sum_net_ticks", lambda x: int((x > 0).sum())),
    )
    by_vol["pct_dates_positive"] = (
        100.0 * by_vol["n_dates_positive"] / by_vol["n_dates"]
    ).round(1)
    by_vol["sum_net_ticks"] = by_vol["sum_net_ticks"].round(2)
    print(by_vol.to_string())
    print()

    # HC #271(A) deployment gate verdicts
    total_net = float(merged["sum_net_ticks"].sum())
    print("=" * 50)
    print("HC #271(A) DEPLOYMENT GATE")
    print("=" * 50)
    print(f"Total NET ticks across all folds: {total_net:+.2f}")

    by_trend_net = merged.groupby("trend_label")["sum_net_ticks"].sum()
    print(f"Per-trend NET: {by_trend_net.to_dict()}")
    profitable_regimes = (by_trend_net > 0).sum()
    print(f"  Profitable in ≥1 trend regime? {'YES' if profitable_regimes >= 1 else 'NO'} "
          f"({profitable_regimes}/3)")

    if total_net != 0:
        max_share = by_trend_net.abs().max() / abs(total_net) if total_net != 0 else 0.0
        print(f"  Max single-trend |NET| / |total NET|: {max_share:.1%} "
              f"{'PASS' if max_share <= 0.5 else 'FAIL'} (HC #271(A) ceiling 50%)")
    else:
        print("  Total NET is zero, can't compute concentration ratio.")

    n_with_trades = int((merged["n_trades"] > 0).sum())
    n_pos = int((merged["sum_net_ticks"] > 0).sum())
    if n_with_trades > 0:
        pct_pos = 100.0 * n_pos / n_with_trades
        print(f"  % folds with trades + positive NET: {pct_pos:.1f}% "
              f"({'PASS' if pct_pos >= 60 else 'FAIL'} HC #254 ≥60% floor)")

    print()
    print(f"WROTE {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
