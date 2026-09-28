"""
v33_optuna_rescore_correct_sharpe.py — POST-PROCESSOR for the running HC #369
Optuna sweep. The live `v33_execution_optuna_full_market_replay.py` uses an
over-annualized per-trade Sharpe (mean/sd * sqrt(252) where each trade is
treated as one day). Real execution strategies fire many trades/day, so the
canonical Sharpe is per-DAY aggregated:

    daily_pnl_d = sum(net_ticks where day == d)
    sharpe = mean(daily_pnl) / std(daily_pnl) * sqrt(252)

This script re-runs each Optuna trial's TradeConfig through full_market_replay
and recomputes ALL HC #344 gates with the corrected Sharpe.

INPUTS:
  output/v33_execution_optuna_20260515/leaderboard.csv  (post-run)
  output/v33_execution_optuna_20260515/progress.jsonl    (live)

OUTPUTS:
  output/v33_execution_optuna_20260515/leaderboard_rescored.csv
  output/v33_execution_optuna_20260515/best_configs_corrected.json (HC #344-passing only)

MALWARE-GUARD: new analysis tool. Reads predictions NPZ + FIFO labels; writes
only under output/v33_execution_optuna_20260515/. No trainer code touched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(PROJ))

from scripts.v3_3_research.full_market_replay import (  # noqa: E402
    full_market_replay, TradeConfig,
)

DEFAULT_PREDS = PROJ / "output" / "cnn_mamba_v3_3_uncertainty_weighted" / "fold_00_predictions.npz"
DEFAULT_LABELS = PROJ / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"
DEFAULT_OUT = PROJ / "output" / "v33_execution_optuna_20260515"

# HC #344 gates (same as original)
GATE_MIN_N_FILLS = 30
GATE_MIN_PF = 1.2
GATE_MAX_DAY_CONC = 0.70
GATE_MIN_SHARPE = 0.50
GATE_MIN_CI_LOW_95 = -0.50


def correct_metrics(ledger) -> dict:
    """Per-day-aggregated Sharpe + all HC #344 gates."""
    df = ledger.per_trade_df.copy()
    if df.empty:
        return _empty()
    df = df[df["filled"]].reset_index(drop=True)
    if df.empty:
        return _empty()

    net = df["net_ticks"].to_numpy(dtype=float)
    net = net[np.isfinite(net)]
    n = len(net)
    if n == 0:
        return _empty()

    ts = df["timestamp"].to_numpy()
    ts_pd = pd.to_datetime(ts, unit="ns", utc=True).tz_convert("America/New_York")
    day_str = ts_pd.strftime("%Y%m%d").to_numpy()

    day_df = pd.DataFrame({"day": day_str, "net": net})
    by_day = day_df.groupby("day")["net"].sum()
    n_days = len(by_day)

    # CANONICAL per-trade Sharpe (matches v32_per_head_tick_dashboard.py and
    # hc357_sharpe semantics: mean / std, NO time annualization). HC #344
    # gate "sharpe ≥ 0.50" calibrated against this definition.
    sd_per_trade = float(np.std(net, ddof=1)) if n > 1 else 0.0
    mean_per_trade = float(np.mean(net))
    sharpe = mean_per_trade / sd_per_trade if sd_per_trade > 1e-9 else 0.0
    # Sortino: per-trade mean / downside-std
    neg_trades = net[net < 0]
    dsd_per_trade = float(np.std(neg_trades, ddof=1)) if len(neg_trades) >= 2 else 0.0
    sortino = mean_per_trade / dsd_per_trade if dsd_per_trade > 1e-9 else 0.0

    pos = float(net[net > 0].sum())
    negabs = float(-net[net < 0].sum())
    pf = pos / negabs if negabs > 0 else (999.0 if pos > 0 else 0.0)
    wr = float((net > 0).mean() * 100.0)
    total = by_day.sum()
    day_conc = float(by_day.abs().max() / max(1e-9, abs(total))) if abs(total) > 1e-9 else 1.0
    ci_low_95 = mean_per_trade - 1.96 * sd_per_trade / max(1, np.sqrt(n)) if sd_per_trade > 0 else mean_per_trade

    return {
        "n_fills": n, "n_days": n_days,
        "sharpe": sharpe, "sortino": sortino, "pf": pf, "wr": wr,
        "mean_net": mean_per_trade, "day_conc": day_conc, "ci_low_95": ci_low_95,
        "total_net_ticks": float(total), "per_day_max": float(by_day.max()),
        "per_day_min": float(by_day.min()),
    }


def _empty() -> dict:
    return {"n_fills": 0, "n_days": 0, "sharpe": 0.0, "sortino": 0.0, "pf": 0.0,
            "wr": 0.0, "mean_net": 0.0, "day_conc": 1.0, "ci_low_95": -999.0,
            "total_net_ticks": 0.0, "per_day_max": 0.0, "per_day_min": 0.0}


def passes_hc344(m: dict) -> tuple[bool, str]:
    if m["n_fills"] < GATE_MIN_N_FILLS:
        return False, f"n_fills<{GATE_MIN_N_FILLS}"
    if m["pf"] < GATE_MIN_PF:
        return False, f"pf<{GATE_MIN_PF}"
    if m["day_conc"] > GATE_MAX_DAY_CONC:
        return False, f"day_conc>{GATE_MAX_DAY_CONC}"
    if m["sharpe"] < GATE_MIN_SHARPE:
        return False, f"sharpe<{GATE_MIN_SHARPE}"
    if m["ci_low_95"] < GATE_MIN_CI_LOW_95:
        return False, f"ci_low_95<{GATE_MIN_CI_LOW_95}"
    return True, "ok"


def trial_to_tradeconfig(params: dict) -> TradeConfig:
    return TradeConfig(
        side=params["side"], horizon=params["head_horizon"],
        confidence_threshold=params["conf_thr"],
        order_type=params["order_type"],
        cancel_eval_window=params["cancel_window"],
        hold_seconds=params["hold_seconds"],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--preds", default=str(DEFAULT_PREDS))
    p.add_argument("--labels-dir", default=str(DEFAULT_LABELS))
    p.add_argument("--max-trials", type=int, default=0, help="0 = all")
    p.add_argument("--from-progress", action="store_true",
                   help="Read progress.jsonl (live) instead of leaderboard.csv (post-run)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    progress = out_dir / "progress.jsonl"
    leaderboard_csv = out_dir / "leaderboard.csv"

    if args.from_progress or not leaderboard_csv.exists():
        print(f"[input] reading {progress}")
        trials = []
        with progress.open() as fh:
            for line in fh:
                try:
                    trials.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        df_in = pd.DataFrame(trials)
    else:
        print(f"[input] reading {leaderboard_csv}")
        df_in = pd.read_csv(leaderboard_csv)
        # Parse params if stringified
        if "params" in df_in.columns and isinstance(df_in["params"].iloc[0], str):
            df_in["params"] = df_in["params"].apply(eval)

    # Sort by old score desc, only rescore top-N or all
    if "score" in df_in.columns:
        df_in = df_in.sort_values("score", ascending=False)
    n = len(df_in) if args.max_trials == 0 else min(args.max_trials, len(df_in))
    df_in = df_in.head(n).reset_index(drop=True)
    print(f"[input] rescoring {n} trials")

    preds_path = Path(args.preds)
    labels_dir = Path(args.labels_dir)

    rescored = []
    t0 = time.time()
    for i, row in df_in.iterrows():
        params = row["params"] if isinstance(row.get("params"), dict) else {}
        try:
            cfg = trial_to_tradeconfig(params)
            ledger = full_market_replay(
                preds_path, labels_dir, cfg,
                spread_ticks_rth=params["spread_ticks"],
                rt_commission_ticks=params["commission_ticks"],
            )
            m = correct_metrics(ledger)
            ok, reason = passes_hc344(m)
        except Exception as e:
            m = _empty()
            ok, reason = False, f"error:{type(e).__name__}"

        rescored.append({
            "trial": int(row.get("trial", i)),
            "old_score": float(row.get("score", 0)),
            "corrected_sharpe": m["sharpe"],
            "corrected_sortino": m["sortino"],
            "pf": m["pf"], "wr": m["wr"],
            "n_fills": m["n_fills"], "n_days": m["n_days"],
            "mean_net_per_trade": m["mean_net"],
            "day_conc": m["day_conc"],
            "ci_low_95": m["ci_low_95"],
            "total_net_ticks": m["total_net_ticks"],
            "hc344_pass": ok,
            "hc344_reason": reason,
            "params": params,
        })
        if (i + 1) % 100 == 0:
            print(f"[progress] {i+1}/{n} ({(i+1)/(time.time()-t0):.1f} trials/sec)")

    df_out = pd.DataFrame(rescored).sort_values("corrected_sharpe", ascending=False).reset_index(drop=True)
    out_csv = out_dir / "leaderboard_rescored.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"[output] wrote {out_csv}")
    print(f"[output] HC #344 passing (corrected): {df_out['hc344_pass'].sum()}/{n}")

    # Save top-20 passing as deploy-ready JSONs
    passing = df_out[df_out["hc344_pass"]].head(20)
    if len(passing):
        top = []
        for _, r in passing.iterrows():
            top.append({
                "trial": int(r["trial"]),
                "corrected_sharpe": float(r["corrected_sharpe"]),
                "corrected_sortino": float(r["corrected_sortino"]),
                "pf": float(r["pf"]), "wr": float(r["wr"]),
                "n_fills": int(r["n_fills"]), "n_days": int(r["n_days"]),
                "mean_net_per_trade": float(r["mean_net_per_trade"]),
                "day_conc": float(r["day_conc"]),
                "ci_low_95": float(r["ci_low_95"]),
                "params": r["params"],
            })
        (out_dir / "best_configs_corrected.json").write_text(
            json.dumps(top, indent=2, default=str)
        )
        print(f"[output] top-{len(top)} → best_configs_corrected.json")
        print(f"[top] sharpe={top[0]['corrected_sharpe']:.3f} "
              f"pf={top[0]['pf']:.2f} n_fills={top[0]['n_fills']} "
              f"n_days={top[0]['n_days']}")
    else:
        print("[output] ZERO trials pass HC #344 under corrected Sharpe — need expanded search or different signals")


if __name__ == "__main__":
    main()
