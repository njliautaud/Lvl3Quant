#!/usr/bin/env python3
"""
HC #397B canonical-replay re-eval of trained PPO v3 (HC #399 canonical-reward) policy.

This is a direct port of ppo_v2_1_canonical_replay_eval.py to v3. Differences:
  - Imports CanonicalReplayEnv from env_v3_canonical (not V33SmartExecEnvV2)
  - v3 env IS canonical replay: each _TradeEvent already carries canon_net_ticks
    and canon_adv_sel_30s. No external canonical_reprice() step needed.
  - Per-day episode loop (set episode_day_idx) instead of random reset start —
    deterministically covers every held-out OOT day exactly once.
  - 7 actions (adds EXIT_POS) instead of 6.
  - Output CSVs: ppo_v3_canonical_replay.csv + ppo_v3_canonical_replay_raw_ledger.csv
  - MLflow experiment: RL_v3_3_smart_exec_v3_canonical_reward (matches training).

Everything else (summarize, rules baseline, HC #392 cost basis, HC #344
day_conc gate, action-distribution telemetry) is IDENTICAL to v2.1 for
apples-to-apples comparison.

NOT MALWARE. Pure analysis script. Read-only on weights and data.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3))
sys.path.insert(0, str(LVL3 / "scripts" / "rl_v3_3_smart_exec"))

from stable_baselines3 import PPO

from env_v3_canonical import (  # noqa: E402
    CanonicalReplayEnv,
    CANCEL_EVAL_WINDOW,
    COMMISSION_RT_TICKS,
    N_ACTIONS,
    A_HOLD, A_BID, A_ASK, A_MKT_BUY, A_MKT_SELL, A_CANCEL, A_EXIT_POS,
)

from scripts.v3_3_research.full_market_replay import (  # noqa: E402
    PRICE_UNIT_TO_TICKS,
    full_market_replay,
    TradeConfig,
)

PREDS = LVL3 / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
LABELS_DIR = LVL3 / "data/processed/mbo_events_smart_v3_fifo_labels"
MODEL_PATH = LVL3 / "output/rl_v3_3_smart_exec_v3/ppo_v3_canonical_final.zip"
OUT_DIR = LVL3 / "output/rl_v3_3_smart_exec_v3"
OUT_CSV = OUT_DIR / "ppo_v3_canonical_replay.csv"
V21_CSV = LVL3 / "output/rl_v3_3_smart_exec/ppo_v2_1_canonical_replay.csv"

RT_COMM = COMMISSION_RT_TICKS  # 0.376
DAY_CONC_GATE = 0.20


# -----------------------------------------------------------------------------
# Risk-adjusted metrics (IDENTICAL to v2.1 summarize())
# -----------------------------------------------------------------------------
def summarize(net: np.ndarray, ts_ns: np.ndarray | None, label: str) -> dict:
    n_filled = int(net.size)
    if n_filled == 0:
        return {
            "name": label, "n_trades": 0, "ticks_per_trade": 0.0, "ticks_total": 0.0,
            "sharpe_sqrtN": 0.0, "sortino_sqrtN": 0.0, "profit_factor": 0.0,
            "win_rate": 0.0, "max_dc_ticks": 0.0, "adv_sel_30s_avg": 0.0,
            "day_conc": float("nan"), "pass_hc344": False, "fills": 0,
            "cancel_window": CANCEL_EVAL_WINDOW, "avg_queue_pos": float("nan"),
        }
    total = float(net.sum())
    mean_ = float(net.mean())
    sd_ = float(net.std(ddof=1)) if n_filled >= 2 and net.std(ddof=1) > 1e-12 else float("nan")
    sharpe = (mean_ / sd_) * np.sqrt(n_filled) if np.isfinite(sd_) else float("nan")
    neg = net[net < 0]
    if neg.size >= 2 and neg.std(ddof=1) > 1e-12:
        sortino = (mean_ / float(neg.std(ddof=1))) * np.sqrt(n_filled)
    else:
        sortino = float("inf") if mean_ > 0 else float("nan")
    gw = float(net[net > 0].sum())
    gl = -float(net[net < 0].sum())
    pf = (gw / gl) if gl > 1e-12 else (float("inf") if gw > 0 else float("nan"))
    wr = float((net > 0).mean() * 100.0)
    eq = np.cumsum(net)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    max_dc = float(dd.max()) if dd.size else 0.0
    if ts_ns is not None and len(ts_ns) == n_filled:
        try:
            dts = pd.to_datetime(ts_ns, unit="ns", utc=True).tz_convert("America/Chicago").date
            df = pd.DataFrame({"date": dts, "net": net})
            per_day = df.groupby("date")["net"].sum()
            total_abs = per_day.abs().sum()
            day_conc = float(per_day.abs().max() / total_abs) if total_abs > 1e-12 else float("nan")
        except Exception:
            day_conc = float("nan")
    else:
        day_conc = float("nan")
    pass_hc344 = bool(np.isfinite(day_conc) and day_conc <= DAY_CONC_GATE and n_filled >= 30)
    return {
        "name": label, "n_trades": n_filled,
        "ticks_per_trade": total / max(1, n_filled), "ticks_total": total,
        "sharpe_sqrtN": sharpe, "sortino_sqrtN": sortino,
        "profit_factor": pf, "win_rate": wr,
        "max_dc_ticks": max_dc, "day_conc": day_conc,
        "pass_hc344": pass_hc344, "fills": n_filled,
        "cancel_window": CANCEL_EVAL_WINDOW,
        "avg_queue_pos": float("nan"),  # populated by caller for canon rows
    }


# -----------------------------------------------------------------------------
# PPO rollout — deterministic per-day coverage of the OOT holdout
# -----------------------------------------------------------------------------
def rollout_ppo_v3(model_path: Path, npz_path: Path, labels_dir: Path,
                   holdout_frac: float, seed: int):
    print(f"[ppo] loading {model_path}")
    model = PPO.load(str(model_path), device="cpu")

    # First, peek at oot_dates to determine holdout day indices
    probe_env = CanonicalReplayEnv(
        npz_path=str(npz_path), labels_dir=str(labels_dir),
        seed=seed, episode_day_idx=0, deterministic_queue=True,
    )
    n_days = probe_env._n_days
    n_hold = max(1, int(math.ceil(n_days * holdout_frac)))
    hold_day_indices = list(range(n_days - n_hold, n_days))
    held_dates = [probe_env.oot_dates[i] for i in hold_day_indices]
    print(f"[ppo] n_days={n_days} n_hold={n_hold} held_indices={hold_day_indices} "
          f"held_dates={held_dates}")

    # Reuse probe_env across days by overriding episode_day_idx per reset.
    env = probe_env

    all_events = []
    action_counts = np.zeros(N_ACTIONS, dtype=np.int64)
    total_steps = 0

    for di in hold_day_indices:
        env._episode_day_override = di
        obs, info = env.reset(seed=seed + di)
        terminated = truncated = False
        ep_steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            a = int(action)
            action_counts[a] += 1
            obs, reward, terminated, truncated, info = env.step(a)
            ep_steps += 1
        # Pull this episode's events with day_idx tag
        for ev in env.events:
            all_events.append({
                "day_idx": di,
                "day_date": env.oot_dates[di],
                "entry_step_global": int(ev.entry_step),
                "exit_step_global": int(getattr(ev, "exit_step", -1) or -1),
                "side": int(ev.side),
                "action_type": str(ev.action_type),
                "filled": bool(getattr(ev, "filled", False)),
                "canon_net_ticks": float(getattr(ev, "canon_net_ticks", 0.0)),
                "canon_adv_sel_30s": float(getattr(ev, "canon_adv_sel_30s", 0.0)),
            })
        total_steps += ep_steps
        n_trades_ep = len(env.events)
        n_filled_ep = sum(1 for ev in env.events if getattr(ev, "filled", False))
        print(f"[ppo] day {di} ({env.oot_dates[di]}): "
              f"steps={ep_steps} events={n_trades_ep} filled={n_filled_ep}")

    print(f"[ppo] rollout done — total_steps={total_steps} total_events={len(all_events)}")
    names = ["HOLD", "BID", "ASK", "MKT_BUY", "MKT_SELL", "CANCEL", "EXIT_POS"]
    print(f"[ppo] action_counts ({','.join(names)}) = {action_counts.tolist()}")

    # Map entry_step → ts_ns via env.fifo
    ts_ns_arr = env.fifo["ts_ns"]
    for ev in all_events:
        ei = ev["entry_step_global"]
        ev["ts_ns"] = int(ts_ns_arr[ei]) if 0 <= ei < len(ts_ns_arr) else -1

    return all_events, action_counts, held_dates


# -----------------------------------------------------------------------------
# Canonical rules baseline (IDENTICAL to v2.1)
# -----------------------------------------------------------------------------
def canonical_rules_baseline(npz_path, labels_dir, held_dates):
    config = TradeConfig(
        side="short", horizon="5s", confidence_threshold=0.005,
        order_type="passive_at_touch", cancel_eval_window=40, hold_seconds=10.0,
    )
    print(f"[rules] held_dates={held_dates}")
    ledger = full_market_replay(
        predictions_npz_path=npz_path, mbo_labels_dir=labels_dir,
        config=config, dates=held_dates, verbose=True,
    )
    net = np.asarray(
        ledger.per_trade_df.loc[ledger.per_trade_df["filled"], "net_ticks"].values,
        dtype=np.float64,
    )
    net = net[np.isfinite(net)]
    ts_filled = ledger.per_trade_df.loc[ledger.per_trade_df["filled"], "timestamp"].values
    ts_filled = np.asarray(ts_filled, dtype=np.int64)
    row = summarize(net, ts_filled, label="rules_j6_top005_short_passive_hold10s_CANON_v3")
    row["adv_sel_30s_avg"] = float(ledger.adverse_selection_cost_ticks_avg)
    row["cancel_window"] = 40  # rules config
    return row


# -----------------------------------------------------------------------------
# v2.1 reference row (for context, read from existing v2.1 CSV)
# -----------------------------------------------------------------------------
def v21_ref_row():
    if not V21_CSV.exists():
        return {"name": "ppo_v2_1_canonical_REF_missing", "n_trades": 0,
                "fills": 0, "ticks_per_trade": float("nan"),
                "ticks_total": float("nan"), "sharpe_sqrtN": float("nan"),
                "sortino_sqrtN": float("nan"), "profit_factor": float("nan"),
                "win_rate": float("nan"), "max_dc_ticks": float("nan"),
                "adv_sel_30s_avg": float("nan"), "day_conc": float("nan"),
                "pass_hc344": False, "cancel_window": 50,
                "avg_queue_pos": float("nan")}
    df = pd.read_csv(V21_CSV)
    cand = df[df["name"] == "ppo_v3_3_v2_canonical_replay"]
    if cand.empty:
        cand = df.head(1)
    r = cand.iloc[0].to_dict()
    r["name"] = "ppo_v2_1_canonical_REF"
    return r


# -----------------------------------------------------------------------------
def log_mlflow(uri, experiment, run_name, rows, args_dict, action_counts):
    try:
        import mlflow
    except ImportError:
        print("[mlflow] not available, skipping")
        return None
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        run = mlflow.start_run(run_name=run_name)
        rid = run.info.run_id
        mlflow.log_params({f"arg_{k}": str(v) for k, v in args_dict.items()})
        total = float(max(1, action_counts.sum()))
        names = ["hold", "bid", "ask", "mkt_buy", "mkt_sell", "cancel", "exit_pos"]
        for nm, c in zip(names, action_counts):
            mlflow.log_metric(f"action_{nm}_count", float(c))
            mlflow.log_metric(f"action_{nm}_pct", 100.0 * float(c) / total)
        for r in rows:
            prefix = (
                "ppo_v3_canon" if r["name"].startswith("ppo_v3_canonical")
                else "rules_canon" if r["name"].startswith("rules_")
                else "ppo_v2_1_ref" if r["name"].startswith("ppo_v2_1")
                else "other"
            )
            for k, v in r.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    mlflow.log_metric(f"{prefix}_{k}", float(v))
                elif isinstance(v, bool):
                    mlflow.log_metric(f"{prefix}_{k}", 1.0 if v else 0.0)
        mlflow.end_run()
        print(f"[mlflow] logged run {rid}")
        return rid
    except Exception as e:
        print(f"[mlflow] failed: {e}", file=sys.stderr)
        return None


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--npz", default=str(PREDS))
    ap.add_argument("--labels-dir", default=str(LABELS_DIR))
    ap.add_argument("--output-csv", default=str(OUT_CSV))
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mlflow-uri", default="http://localhost:5000")
    ap.add_argument("--mlflow-experiment",
                    default="RL_v3_3_smart_exec_v3_canonical_reward")
    args = ap.parse_args()

    print(f"[main] start @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[main] args = {vars(args)}")

    all_events, action_counts, held_dates = rollout_ppo_v3(
        model_path=Path(args.model), npz_path=Path(args.npz),
        labels_dir=Path(args.labels_dir),
        holdout_frac=args.holdout_frac, seed=args.seed,
    )

    if not all_events:
        print("[ERROR] PPO v3 produced ZERO trade events. Cannot summarize.")
        total = float(max(1, action_counts.sum()))
        names = ["HOLD", "BID", "ASK", "MKT_BUY", "MKT_SELL", "CANCEL", "EXIT_POS"]
        for nm, c in zip(names, action_counts):
            print(f"  {nm:<10} {int(c):>10}  ({100.0*c/total:.2f}%)")
        sys.exit(2)

    # Raw ledger CSV
    raw_ledger_csv = OUT_DIR / "ppo_v3_canonical_replay_raw_ledger.csv"
    pd.DataFrame(all_events).to_csv(raw_ledger_csv, index=False)
    print(f"[main] wrote raw ledger: {raw_ledger_csv}")

    # Extract filled-trade net + ts for summary
    df_ev = pd.DataFrame(all_events)
    filled_df = df_ev[df_ev["filled"]].copy()
    n_attempts = int(len(df_ev))
    n_filled = int(len(filled_df))
    n_market = int((df_ev["action_type"] == "market").sum())
    n_passive = int((df_ev["action_type"] == "passive").sum())
    print(f"[canon] events={n_attempts} (market={n_market}, passive={n_passive}) "
          f"filled={n_filled}")

    if n_filled == 0:
        net_filled = np.array([], dtype=np.float64)
        ts_filled = np.array([], dtype=np.int64)
        adv_sel_avg = 0.0
    else:
        net_filled = filled_df["canon_net_ticks"].to_numpy(dtype=np.float64)
        ts_filled = filled_df["ts_ns"].to_numpy(dtype=np.int64)
        adv_sel_avg = float(filled_df["canon_adv_sel_30s"].mean())

    # "All attempts" row: net=0 for unfilled (HC #397B comparable to v2.1 all-row)
    net_all = df_ev["canon_net_ticks"].to_numpy(dtype=np.float64)
    ts_all = df_ev["ts_ns"].to_numpy(dtype=np.int64)
    row_canon = summarize(net_all, ts_all, label="ppo_v3_canonical_replay")
    row_canon["adv_sel_30s_avg"] = adv_sel_avg
    row_canon["fills"] = n_filled
    row_canon["avg_queue_pos"] = float("nan")  # env uses parametric queue model;
    # avg position not directly exposed — cancel_window=50 documents the config
    print(f"[main] canon PPO v3 (all-attempts): {row_canon}")

    row_canon_fills = summarize(net_filled, ts_filled,
                                label="ppo_v3_canonical_replay_fills_only")
    row_canon_fills["adv_sel_30s_avg"] = adv_sel_avg
    row_canon_fills["fills"] = n_filled
    row_canon_fills["avg_queue_pos"] = float("nan")
    print(f"[main] canon PPO v3 (fills-only): {row_canon_fills}")

    row_rules = canonical_rules_baseline(
        npz_path=Path(args.npz), labels_dir=Path(args.labels_dir),
        held_dates=held_dates,
    )
    print(f"[main] canon rules: {row_rules}")

    row_v21_ref = v21_ref_row()
    print(f"[main] v2.1 ref: {row_v21_ref}")

    # Action distribution row (HOLD,BID,ASK,MKT_BUY,MKT_SELL,CANCEL,EXIT_POS)
    total = float(max(1, action_counts.sum()))
    action_row = {
        "name": "ppo_v3_action_distribution_PCT",
        "n_trades": int(action_counts.sum()),
        "fills": 0,
        "ticks_per_trade": 100.0 * action_counts[0] / total,    # HOLD%
        "ticks_total": 100.0 * action_counts[1] / total,        # BID%
        "sharpe_sqrtN": 100.0 * action_counts[2] / total,       # ASK%
        "sortino_sqrtN": 100.0 * action_counts[3] / total,      # MKT_BUY%
        "profit_factor": 100.0 * action_counts[4] / total,      # MKT_SELL%
        "win_rate": 100.0 * action_counts[5] / total,           # CANCEL%
        "max_dc_ticks": 100.0 * action_counts[6] / total,       # EXIT_POS% (new in v3)
        "adv_sel_30s_avg": float("nan"),
        "day_conc": float("nan"),
        "pass_hc344": False,
        "cancel_window": CANCEL_EVAL_WINDOW,
        "avg_queue_pos": float("nan"),
    }

    fieldnames = [
        "name", "n_trades", "fills", "ticks_per_trade", "ticks_total",
        "sharpe_sqrtN", "sortino_sqrtN", "profit_factor", "win_rate",
        "max_dc_ticks", "adv_sel_30s_avg", "day_conc", "pass_hc344",
        "cancel_window", "avg_queue_pos",
    ]
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in (row_canon, row_canon_fills, row_rules, row_v21_ref, action_row):
            for k in fieldnames:
                r.setdefault(k, float("nan"))
            w.writerow(r)
    print(f"[main] wrote CSV: {args.output_csv}")

    rid = log_mlflow(
        uri=args.mlflow_uri, experiment=args.mlflow_experiment,
        run_name="ppo_v3_canonical_eval",
        rows=[row_canon, row_canon_fills, row_rules, row_v21_ref],
        args_dict=vars(args), action_counts=action_counts,
    )

    print("\n========== HC #397B CANONICAL COMPARISON — PPO v3 ==========")
    print(f"{'name':<55} {'tk/tr':>8} {'fills':>6} {'Shrp':>8} {'PF':>6} {'WR%':>6} "
          f"{'advS':>7} {'dayC':>6} {'pass':>5}")
    for r in (row_canon, row_canon_fills, row_rules, row_v21_ref):
        def f(k, fmt=".3f"):
            v = r.get(k, float("nan"))
            if isinstance(v, bool):
                return str(v)
            try:
                return f"{float(v):{fmt}}"
            except Exception:
                return "nan"
        print(f"{r['name']:<55} {f('ticks_per_trade'):>8} "
              f"{int(r.get('fills',0) or 0):>6} {f('sharpe_sqrtN','.2f'):>8} "
              f"{f('profit_factor','.2f'):>6} {f('win_rate','.1f'):>6} "
              f"{f('adv_sel_30s_avg','.3f'):>7} "
              f"{f('day_conc','.3f'):>6} {str(r.get('pass_hc344','?')):>5}")
    names = ["HOLD", "BID", "ASK", "MKT_BUY", "MKT_SELL", "CANCEL", "EXIT_POS"]
    print("\n========== ACTION DISTRIBUTION (PPO v3) ==========")
    for nm, c in zip(names, action_counts):
        print(f"  {nm:<10} {int(c):>10}  ({100.0*c/total:.2f}%)")
    print(f"\nmlflow_run_id: {rid}")
    print(f"[main] DONE @ {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
