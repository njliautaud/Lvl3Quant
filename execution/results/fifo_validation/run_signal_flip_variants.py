#!/usr/bin/env python3
"""
CNN-Mamba v2 — Signal-Flip Exit Variant Sweep
==============================================
User direction (Apr 27 2026): "Continue to optimize execution for the CNN Mamba v2."

Builds on vol-exit production config (chase + SL15 + vol_exit 5/5b + 60s + prime hours)
and tests dynamic signal-flip exit variants the user asked about.

Variants tested:
  baseline : conv=20 mag=0.5 hold=60s   (current vol-exit production — control)
  A_inst2  : conv=1  mag=2.0 hold=60s   (instant exit on opposite z>=2)
  B_5b15   : conv=5  mag=1.5 hold=60s   (500ms sustained at |z|>=1.5)
  D_noTS   : conv=20 mag=0.5 hold=∞     (kill the 60s time stop)
  E_3b10   : conv=3  mag=1.0 hold=60s   (300ms sustained at |z|>=1)
  F_AnoTS  : conv=1  mag=2.0 hold=∞     (instant flip + no time stop)
  G_inst   : signal_flip_exit=true conv=0  hold=60s (any opposite, instant)

Z thresholds: 2.0, 2.5, 3.0  (the profitable tier from prior sweeps)
Dates:        Mar 2-5 (validation set, 4 days)
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path

LVL3 = Path("/home/jupiter/Lvl3Quant")
BINARY = LVL3 / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
PRED_DIR = LVL3 / "execution" / "results" / "fifo_validation" / "pred_cache" / "cnn_mamba_v2"
MBO_DIR = LVL3 / "data" / "raw" / "mbo"
RESULTS_DIR = LVL3 / "execution" / "results" / "fifo_validation"

DATES = ["20260302", "20260303", "20260304", "20260305"]
THRESHOLDS = [2.0, 2.5, 3.0]

# Common base flags applied to ALL variants
BASE_ARGS = [
    "--chase-entry",
    "--chase-max-ticks", "2",
    "--chase-max-reprices", "5",
    "--stop-loss-ticks", "15",
    "--vol-exit-ticks", "5",
    "--vol-exit-bars", "5",
    "--latency-ms", "5",
    "--prime-hours",
    "--quiet",
]

# Variant-specific overrides
VARIANTS = {
    "baseline": {
        "hold_ms": "60000",
        "conviction_exit_bars": "20",
        "conviction_exit_mag": "0.5",
        "signal_flip_exit": False,
    },
    "A_inst2": {
        "hold_ms": "60000",
        "conviction_exit_bars": "1",
        "conviction_exit_mag": "2.0",
        "signal_flip_exit": False,
    },
    "B_5b15": {
        "hold_ms": "60000",
        "conviction_exit_bars": "5",
        "conviction_exit_mag": "1.5",
        "signal_flip_exit": False,
    },
    "D_noTS": {
        "hold_ms": "9999999",  # ~167 min, effectively no time stop
        "conviction_exit_bars": "20",
        "conviction_exit_mag": "0.5",
        "signal_flip_exit": False,
    },
    "E_3b10": {
        "hold_ms": "60000",
        "conviction_exit_bars": "3",
        "conviction_exit_mag": "1.0",
        "signal_flip_exit": False,
    },
    "F_AnoTS": {
        "hold_ms": "9999999",
        "conviction_exit_bars": "1",
        "conviction_exit_mag": "2.0",
        "signal_flip_exit": False,
    },
    "G_inst": {
        "hold_ms": "60000",
        "conviction_exit_bars": "0",
        "conviction_exit_mag": "0.0",
        "signal_flip_exit": True,
    },
}


def run_sim(variant, date, threshold, out_dir):
    pred = PRED_DIR / f"{date}.npz"
    mbo = MBO_DIR / f"glbx-mdp3-{date}.mbo.dbn.zst"
    tag = f"{variant}_z{threshold:.1f}_{date}"
    out = out_dir / f"{tag}.json"
    v = VARIANTS[variant]
    cmd = [
        str(BINARY),
        "--mbo-file", str(mbo),
        "--predictions", str(pred),
        "--output", str(out),
        "--signal-threshold", str(threshold),
        "--hold-ms", v["hold_ms"],
        "--conviction-exit-bars", v["conviction_exit_bars"],
        "--conviction-exit-mag", v["conviction_exit_mag"],
    ] + BASE_ARGS
    if v["signal_flip_exit"]:
        cmd.append("--signal-flip-exit")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            return {"error": r.stderr[-500:], "tag": tag}
        return json.loads(out.read_text())
    except Exception as e:
        return {"error": str(e), "tag": tag}


def aggregate(per_date):
    """Aggregate metrics across dates for a (variant, z) cell.
    Uses the actual fill_sim_cli output keys.
    """
    valid = [r for r in per_date.values() if isinstance(r, dict) and "error" not in r]
    if not valid:
        return {"total_trades": 0, "win_rate": 0, "profit_factor": 0,
                "total_pnl": 0, "pnl_per_trade": 0,
                "avg_winner": 0, "avg_loser": 0, "n_signals": 0,
                "fill_rate": 0, "sharpe": 0}
    n_trades = sum(r.get("total_trades", 0) for r in valid)
    n_signals = sum(r.get("total_signals", 0) for r in valid)
    n_filled = sum(r.get("total_filled", 0) for r in valid)
    pnl = sum(r.get("total_pnl_dollars", 0) for r in valid)
    # Trade-weighted aggregates
    if n_trades > 0:
        wr = sum(r.get("win_rate", 0) * r.get("total_trades", 0) for r in valid) / n_trades
        pf_per = [r.get("profit_factor", 0) for r in valid if r.get("total_trades", 0) > 0]
        # PF: aggregate properly via gross_profit/gross_loss reconstruction
        # Per fill_sim, profit_factor per day is gross_p/gross_l. We approximate weighted PF via avg_win/avg_loss/win_rate
        n_wins = sum(r.get("win_rate", 0) * r.get("total_trades", 0) for r in valid)
        gp = sum(r.get("avg_win", 0) * r.get("win_rate", 0) * r.get("total_trades", 0) for r in valid)
        gl = sum(abs(r.get("avg_loss", 0)) * (1 - r.get("win_rate", 0)) * r.get("total_trades", 0) for r in valid)
        pf = (gp / gl) if gl > 0 else (999.0 if gp > 0 else 0)
        sharpe = sum(r.get("sharpe_per_trade", 0) * r.get("total_trades", 0) for r in valid) / n_trades
        avg_w = sum(r.get("avg_win", 0) * r.get("win_rate", 0) * r.get("total_trades", 0) for r in valid) / max(n_wins, 1)
        n_loss = n_trades - n_wins
        avg_l = sum(r.get("avg_loss", 0) * (1 - r.get("win_rate", 0)) * r.get("total_trades", 0) for r in valid) / max(n_loss, 1)
    else:
        wr = pf = sharpe = avg_w = avg_l = 0
    return {
        "total_trades": n_trades, "n_signals": n_signals,
        "fill_rate": round(n_filled / max(n_signals, 1), 3),
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 3),
        "total_pnl": round(pnl, 2),
        "pnl_per_trade": round(pnl / max(n_trades, 1), 2),
        "avg_winner": round(avg_w, 2),
        "avg_loser": round(avg_l, 2),
        "sharpe": round(sharpe, 3),
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"sim_signal_flip_variants_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== CNN-Mamba v2 Signal-Flip Variant Sweep ===")
    print(f"Output: {out_dir}")
    print(f"Variants: {list(VARIANTS.keys())}  ({len(VARIANTS)} total)")
    print(f"Z thresholds: {THRESHOLDS}")
    print(f"Dates: {DATES}\n")

    summary = {}
    for variant in VARIANTS:
        for z in THRESHOLDS:
            key = f"{variant}_z{z:.1f}"
            per_date = {}
            for d in DATES:
                r = run_sim(variant, d, z, out_dir)
                per_date[d] = r if isinstance(r, dict) and "error" not in r else {"error": r.get("error", "?")}
            agg = aggregate(per_date)
            summary[key] = agg
            mark = "WIN" if agg["total_pnl"] > 0 and agg["profit_factor"] > 1.1 else ("ok " if agg["total_pnl"] > 0 else "BAD")
            print(f"{mark} {key:<18} n={agg['total_trades']:>4} sig={agg['n_signals']:>5} "
                  f"fill={agg['fill_rate']:.1%} WR={agg['win_rate']:.1%} "
                  f"PF={agg['profit_factor']:>5.2f} PnL=${agg['total_pnl']:>+8.0f} "
                  f"$/t={agg['pnl_per_trade']:>+6.2f} W={agg['avg_winner']:>+6.2f} L={agg['avg_loser']:>+6.2f}", flush=True)

    summary_path = out_dir / "_summary.json"
    summary_path.write_text(json.dumps({
        "variants": VARIANTS, "thresholds": THRESHOLDS, "dates": DATES,
        "summary": summary,
    }, indent=2, default=str))

    print(f"\n=== TOP 5 BY PROFIT FACTOR (min 100 trades) ===")
    rank = sorted(
        [(k, v) for k, v in summary.items() if v["total_trades"] >= 100],
        key=lambda kv: kv[1]["profit_factor"], reverse=True
    )[:5]
    for k, v in rank:
        print(f"  {k:<18} PF={v['profit_factor']:>5.2f} WR={v['win_rate']:.1%} PnL=${v['total_pnl']:>+8.0f} n={v['total_trades']}")

    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
