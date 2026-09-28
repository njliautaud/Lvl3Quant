#!/usr/bin/env python3
"""
Compute Sortino ratio + bootstrap CIs for every candidate config in
configs/winning_configs/. Updates each JSON in place with risk metrics block.

Sortino convention used (per-trade):
    Sortino = mean(pnl_per_trade) / std(downside_pnl_per_trade)
    where downside_pnl = pnl[pnl < 0]   (MAR = 0)
    Annualized to daily by *sqrt(n_trades_per_day_avg) for sanity reference.

Daily Sortino:
    Sortino_daily = mean(daily_pnl) / std(daily_negative_pnl)

Bootstrap (B=5000, stratified by day):
    Resample DAYS with replacement; for each resample compute the metric.
    Report 5/50/95 percentiles.

Outputs per candidate are written to a sibling key `risk_metrics`:
    {
        "sortino_per_trade":     {"point": ..., "ci_low": ..., "ci_high": ..., "n": ...},
        "sortino_daily":         {"point": ..., "ci_low": ..., "ci_high": ..., "n_days": ...},
        "total_pnl_ci":          {"low": ..., "med": ..., "high": ...},
        "win_rate_ci":           {"low": ..., "med": ..., "high": ...},
        "downside_deviation":    ...,
        "max_daily_drawdown":    ...,
        "n_trades":              ...
    }
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

LVL3 = Path("/home/jupiter/Lvl3Quant")
CONFIG_DIR = LVL3 / "configs" / "winning_configs"
SWEEP_ROOTS = [
    LVL3 / "execution" / "results" / "fifo_validation" / "sim_tp_z_finegrain_9day_20260427_101143",
    LVL3 / "execution" / "results" / "fifo_validation" / "sim_tp8_z25_9day_20260427_100909",
]


def find_per_day_files(strategy_name, dates):
    """strategy_name e.g. 'tp9_z2.3_chase2_5repr_...' -> we just use leading 'tp9_z2.3'"""
    parts = strategy_name.split("_")
    tp_tag = parts[0]   # tp9
    z_tag  = parts[1]   # z2.3
    files = {}
    for sweep_root in SWEEP_ROOTS:
        if not sweep_root.exists():
            continue
        for d in dates:
            f = sweep_root / f"{tp_tag}_{z_tag}_{d}.json"
            if f.exists():
                files[d] = f
    return files


def load_trades(per_day_files):
    """Returns list[(date, list[trade_pnl_dollars])] in date order."""
    by_day = []
    for date in sorted(per_day_files.keys()):
        d = json.loads(per_day_files[date].read_text())
        pnls = [t["pnl_dollars"] for t in d.get("trades", [])]
        by_day.append((date, pnls))
    return by_day


def sortino_per_trade(trade_pnls):
    arr = np.asarray(trade_pnls, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    mean = arr.mean()
    downside = arr[arr < 0]
    if downside.size < 2:
        return float("inf") if mean > 0 else 0.0, 0.0
    dd = downside.std(ddof=1)
    return (mean / dd) if dd > 0 else 0.0, dd


def sortino_daily(by_day):
    daily = np.array([sum(p) for _, p in by_day], dtype=float)
    if daily.size == 0:
        return 0.0, daily
    mean = daily.mean()
    downside = daily[daily < 0]
    if downside.size < 2:
        return float("inf") if mean > 0 else 0.0, daily
    dd = downside.std(ddof=1)
    return (mean / dd) if dd > 0 else 0.0, daily


def bootstrap_metrics(by_day, B=5000, seed=42):
    """Stratified bootstrap by day. Returns dict of percentile arrays."""
    rng = np.random.default_rng(seed)
    n_days = len(by_day)
    if n_days == 0:
        return None

    sortino_pt_dist = []
    sortino_d_dist = []
    total_pnl_dist = []
    wr_dist = []

    for _ in range(B):
        idx = rng.integers(0, n_days, size=n_days)
        resample = [by_day[i] for i in idx]

        all_pnl = np.concatenate([np.asarray(p) for _, p in resample]) if any(p for _, p in resample) else np.array([])
        if all_pnl.size == 0:
            continue
        # Per-trade Sortino
        m = all_pnl.mean()
        dn = all_pnl[all_pnl < 0]
        spt = (m / dn.std(ddof=1)) if dn.size > 1 and dn.std(ddof=1) > 0 else (np.sign(m) * 99.0 if m != 0 else 0.0)
        sortino_pt_dist.append(spt)

        # Daily Sortino
        daily = np.array([sum(p) for _, p in resample])
        dm = daily.mean()
        ddwn = daily[daily < 0]
        sd = (dm / ddwn.std(ddof=1)) if ddwn.size > 1 and ddwn.std(ddof=1) > 0 else (np.sign(dm) * 99.0 if dm != 0 else 0.0)
        sortino_d_dist.append(sd)

        total_pnl_dist.append(daily.sum())
        wr_dist.append(float((all_pnl > 0).mean()))

    def pct(a, q):
        if not a:
            return float("nan")
        return float(np.percentile(np.asarray(a), q))

    return {
        "sortino_per_trade": {
            "p5": pct(sortino_pt_dist, 5),
            "p50": pct(sortino_pt_dist, 50),
            "p95": pct(sortino_pt_dist, 95),
        },
        "sortino_daily": {
            "p5": pct(sortino_d_dist, 5),
            "p50": pct(sortino_d_dist, 50),
            "p95": pct(sortino_d_dist, 95),
        },
        "total_pnl_dollars": {
            "p5": pct(total_pnl_dist, 5),
            "p50": pct(total_pnl_dist, 50),
            "p95": pct(total_pnl_dist, 95),
        },
        "win_rate": {
            "p5": pct(wr_dist, 5),
            "p50": pct(wr_dist, 50),
            "p95": pct(wr_dist, 95),
        },
        "n_bootstrap": B,
    }


def max_daily_drawdown(daily_pnl):
    """Worst peak-to-trough drawdown of cumulative daily PnL."""
    if daily_pnl.size == 0:
        return 0.0
    cum = np.cumsum(daily_pnl)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(dd.max())


def process(cfg_path):
    cfg = json.loads(cfg_path.read_text())
    dates = cfg["validation_dates"]
    strat = cfg["strategy_name"]
    per_day = find_per_day_files(strat, dates)
    if not per_day:
        print(f"  [WARN] no per-day files matched for {cfg_path.name}")
        return None

    by_day = load_trades(per_day)
    all_trades = [pnl for _, pnls in by_day for pnl in pnls]
    n_trades = len(all_trades)
    if n_trades == 0:
        print(f"  [WARN] zero trades for {cfg_path.name}")
        return None

    spt, dd = sortino_per_trade(all_trades)
    sd, daily = sortino_daily(by_day)
    boot = bootstrap_metrics(by_day, B=5000)
    mdd = max_daily_drawdown(daily)

    risk = {
        "computed_at": datetime.utcnow().isoformat() + "Z",
        "n_trades": n_trades,
        "n_days": len(by_day),
        "sortino_per_trade": {"point": float(spt), "downside_deviation_dollars": float(dd)},
        "sortino_daily": {"point": float(sd)},
        "max_daily_drawdown_dollars": mdd,
        "daily_pnl": [float(x) for x in daily],
        "bootstrap": boot,
    }
    cfg["risk_metrics"] = risk
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"  ✓ {cfg_path.name}  n_tr={n_trades:>4} Sortino_trade={spt:+.3f} Sortino_daily={sd:+.3f} MDD=${mdd:,.0f}")
    print(f"     bootstrap (5/50/95): Sortino_d={boot['sortino_daily']['p5']:+.2f}/{boot['sortino_daily']['p50']:+.2f}/{boot['sortino_daily']['p95']:+.2f}"
          f"   PnL=${boot['total_pnl_dollars']['p5']:+,.0f}/${boot['total_pnl_dollars']['p50']:+,.0f}/${boot['total_pnl_dollars']['p95']:+,.0f}")
    return risk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default=str(CONFIG_DIR))
    ap.add_argument("--bootstrap", type=int, default=5000)
    args = ap.parse_args()

    cfgs = sorted(Path(args.config_dir).glob("*.json"))
    print(f"Processing {len(cfgs)} candidate configs in {args.config_dir}")
    for c in cfgs:
        if c.name == "README.md":
            continue
        print(f"\n→ {c.name}")
        try:
            process(c)
        except Exception as e:
            print(f"  [ERROR] {e}")

    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
