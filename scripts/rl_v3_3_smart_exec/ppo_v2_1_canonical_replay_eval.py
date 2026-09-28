#!/usr/bin/env python3
"""
HC #397 canonical-replay re-eval of trained PPO v2 (sized-reward) policy.

This is a direct port of ppo_canonical_replay_eval.py (which evaluated v1)
to v2. The ONLY differences from the v1 script:
  - Imports V33SmartExecEnvV2 from env_v2 instead of V33SmartExecEnv from env
  - HeldOutEnv inherits V33SmartExecEnvV2
  - HeldOutEnv passes sizing_calibrator=None at instantiation (shaping does
    NOT affect actions during deterministic inference; it only mutates the
    reward signal which the policy doesn't see at rollout time. Skipping the
    calibrator avoids a hard torch/file dependency at eval time.)
  - MODEL_PATH points at ppo_v3_3_v2_1_final.zip
  - Output CSVs: ppo_v2_1_canonical_replay.csv +
                 ppo_v2_1_canonical_replay_raw_ledger.csv
  - MLflow run_name: ppo_v3_3_v2_sized_canonical_eval

Everything else (canonical_reprice, summarize, rules baseline, MLflow
schema, HC #392 cost basis, HC #344 day_conc gate) is IDENTICAL to v1
for apples-to-apples comparison.

NOT MALWARE. Pure analysis script. Read-only on weights and data.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3))
sys.path.insert(0, str(LVL3 / "scripts" / "rl_v3_3_smart_exec"))

import gymnasium as gym
from stable_baselines3 import PPO

from env_v2 import (  # noqa: E402
    V33SmartExecEnvV2,
    RTH_STEPS,
    COMMISSION_RT_TICKS,
    A_HOLD, A_BID, A_ASK, A_MKT_BUY, A_MKT_SELL, A_CANCEL,
)

from scripts.v3_3_research.full_market_replay import (  # noqa: E402
    ES_RT_COMMISSION_TICKS_DEFAULT,
    ES_SPREAD_TICKS_RTH_DEFAULT,
    PRICE_UNIT_TO_TICKS,
    ANN_FACTOR_PER_STEP,
    _load_fifo_labels,
    _queue_position_model,
    _entry_price_edge_ticks,
    full_market_replay,
    TradeConfig,
)

PREDS = LVL3 / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
LABELS_DIR = LVL3 / "data/processed/mbo_events_smart_v3_fifo_labels"
MODEL_PATH = LVL3 / "output/rl_v3_3_smart_exec/ppo_v3_3_v2_1_final.zip"
OUT_DIR = LVL3 / "output/rl_v3_3_smart_exec"
OUT_CSV = OUT_DIR / "ppo_v2_1_canonical_replay.csv"
LEGACY_V1_CSV = OUT_DIR / "ppo_canonical_replay.csv"  # for v1-vs-v2 context

RT_COMM = ES_RT_COMMISSION_TICKS_DEFAULT  # 0.376
DAY_CONC_GATE = 0.20


# -----------------------------------------------------------------------------
# Held-out env (mirrors v1 HeldOutEnv exactly; just inherits the v2 parent)
# -----------------------------------------------------------------------------
class HeldOutEnv(V33SmartExecEnvV2):
    def __init__(self, npz_path, seed=0, holdout_frac=0.2):
        # sizing_calibrator=None: shaping only mutates rewards, not obs/action.
        # At deterministic inference the policy never sees rewards. Confirmed by
        # diffing env.py vs env_v2.py: observation_space, action_space, _obs(),
        # and all step()-state transitions are byte-identical. Shaping changes
        # only the `reward` return value in three entry-step branches.
        super().__init__(npz_path=npz_path, seed=seed, sizing_calibrator=None)
        self._holdout_start = int(self.N * (1.0 - holdout_frac))
        if self.N - self._holdout_start < 1000:
            self._holdout_start = max(0, self.N - 1000)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        max_start = max(self._holdout_start + 1, self.N - RTH_STEPS - 1)
        lo = self._holdout_start
        hi = max(lo + 1, max_start)
        self.step_idx = int(self.rng.integers(lo, hi))
        self._reset_episode_state_keep_start()
        return self._obs(), {}


# -----------------------------------------------------------------------------
# Trade-ledger instrumentation wrapper (IDENTICAL to v1)
# -----------------------------------------------------------------------------
class TradeLogWrapper(gym.Wrapper):
    def __init__(self, env: HeldOutEnv):
        super().__init__(env)
        self.trades: list[dict] = []
        self._episode = -1
        self._open_trade: dict | None = None
        self._last_action: int = -1
        # Action distribution telemetry (NEW vs v1 — pure additive instrumentation
        # so we can report degenerate-policy diagnostics without touching env)
        self.action_counts = np.zeros(6, dtype=np.int64)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._episode += 1
        self._open_trade = None
        self._last_action = -1
        return obs, info

    def step(self, action: int):
        e = self.env
        action_int = int(action)
        self.action_counts[action_int] += 1
        prev_side = int(e.position_side)
        prev_pending = int(e.pending_order)

        obs, reward, terminated, truncated, info = self.env.step(action_int)

        new_side = int(e.position_side)

        # OPEN detection
        if prev_side == 0 and new_side != 0:
            entry_idx_global = int(e.entry_idx)
            if action_int in (A_MKT_BUY, A_MKT_SELL):
                atype = "market"
            elif prev_pending != 0:
                atype = "passive"
            else:
                atype = "passive"
            self._open_trade = {
                "episode": self._episode,
                "entry_idx": entry_idx_global,
                "side": new_side,
                "action_type": atype,
                "open_step": int(e.step_idx),
            }

        # CLOSE detection
        if prev_side != 0 and new_side == 0 and self._open_trade is not None:
            self._open_trade["exit_idx"] = int(e.step_idx) - 1
            self._open_trade["env_realized_at_close"] = float(e.realized_pnl_ticks)
            self._open_trade["env_step_reward_at_close"] = float(reward)
            self.trades.append(self._open_trade)
            self._open_trade = None

        self._last_action = action_int
        return obs, reward, terminated, truncated, info


# -----------------------------------------------------------------------------
# Risk-adjusted metrics (IDENTICAL to v1)
# -----------------------------------------------------------------------------
def summarize(net: np.ndarray, ts_ns: np.ndarray | None, label: str) -> dict:
    n_filled = int(net.size)
    if n_filled == 0:
        return {
            "name": label, "n_trades": 0, "ticks_per_trade": 0.0, "ticks_total": 0.0,
            "sharpe_sqrtN": 0.0, "sortino_sqrtN": 0.0, "profit_factor": 0.0,
            "win_rate": 0.0, "max_dc_ticks": 0.0, "adv_sel_30s_avg": 0.0,
            "day_conc": float("nan"), "pass_hc344": False, "fills": 0,
        }
    total = float(net.sum())
    mean_ = float(net.mean())
    sd_ = float(net.std(ddof=1)) if n_filled >= 2 and net.std(ddof=1) > 1e-12 else float("nan")
    if np.isfinite(sd_):
        sharpe = (mean_ / sd_) * np.sqrt(n_filled)
    else:
        sharpe = float("nan")
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
    }


# -----------------------------------------------------------------------------
# PPO rollout
# -----------------------------------------------------------------------------
def rollout_ppo(model_path: Path, npz_path: Path, n_episodes: int,
                holdout_frac: float, seed: int):
    print(f"[ppo] loading {model_path}")
    model = PPO.load(str(model_path), device="cpu")
    base_env = HeldOutEnv(str(npz_path), seed=seed, holdout_frac=holdout_frac)
    print(f"[ppo] N={base_env.N} holdout_start={base_env._holdout_start} "
          f"holdout_size={base_env.N - base_env._holdout_start}")
    env = TradeLogWrapper(base_env)
    total_steps = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_steps += 1
        print(f"[ppo] ep{ep+1}/{n_episodes} cum_trades={len(env.trades)}")
    print(f"[ppo] rollout done — steps={total_steps}, trades={len(env.trades)}")
    print(f"[ppo] action_counts (HOLD,BID,ASK,MKT_BUY,MKT_SELL,CANCEL) = "
          f"{env.action_counts.tolist()}")
    return env.trades, base_env.N, env.action_counts


# -----------------------------------------------------------------------------
# Canonical re-price (IDENTICAL to v1)
# -----------------------------------------------------------------------------
def canonical_reprice(trades, npz_path, labels_dir):
    d = np.load(npz_path, allow_pickle=True)
    n_samples = int(d["n_samples"])
    oot_dates = [str(x) for x in d["oot_dates"]]
    print(f"[canon] n_samples={n_samples} oot_dates={oot_dates}")
    fifo = _load_fifo_labels(labels_dir, oot_dates)
    n_fifo = sum(fifo["_n_per_day"])
    n_total = min(n_samples, n_fifo)
    print(f"[canon] n_total={n_total} (preds∩fifo)")

    tgt_lr_30s = d["target_log_ret_30s"][:n_total].astype(np.float64)
    mask_30s = (d["mask_log_ret_30s"][:n_total].astype(bool)
                & np.isfinite(tgt_lr_30s))
    tgt_lr_1s = d["target_log_ret_1s"][:n_total].astype(np.float64)
    mask_1s = d["mask_log_ret_1s"][:n_total].astype(bool) & np.isfinite(tgt_lr_1s)

    n_trades = len(trades)
    net_per_trade = np.zeros(n_trades, dtype=np.float64)
    filled_per_trade = np.zeros(n_trades, dtype=bool)
    ts_per_trade = np.full(n_trades, -1, dtype=np.int64)
    edge_offsets = np.zeros(n_trades, dtype=np.float64)
    adv_sel_30s_per_trade = np.full(n_trades, np.nan, dtype=np.float64)

    passive_idx_in_trades = []
    passive_entry_idx = []
    passive_side = []

    for ti, tr in enumerate(trades):
        entry_idx = int(tr["entry_idx"])
        side = int(tr["side"])
        if entry_idx < 0 or entry_idx >= n_total:
            continue
        ts_per_trade[ti] = int(fifo["ts_ns"][entry_idx])
        if tr["action_type"] == "market":
            if mask_30s[entry_idx]:
                lr = float(tgt_lr_30s[entry_idx]) * PRICE_UNIT_TO_TICKS
                edge = 0.0
                net_per_trade[ti] = side * lr + edge - RT_COMM
                edge_offsets[ti] = edge
                filled_per_trade[ti] = True
                adv_sel_30s_per_trade[ti] = min(0.0, side * lr)
        elif tr["action_type"] == "passive":
            passive_idx_in_trades.append(ti)
            passive_entry_idx.append(entry_idx)
            passive_side.append(side)

    if passive_idx_in_trades:
        p_idx = np.asarray(passive_entry_idx, dtype=np.int64)
        p_side = np.asarray(passive_side, dtype=np.int32)
        for sgn, side_key in ((+1, "long"), (-1, "short")):
            mask_this = (p_side == sgn)
            if not mask_this.any():
                continue
            idx_this = p_idx[mask_this]
            filled_lbl = fifo[f"tp4sl3_{side_key}_filled"][idx_this]
            exit_reason_lbl = fifo[f"tp4sl3_{side_key}_exit_reason"][idx_this]
            hold_time_lbl = fifo[f"tp4sl3_{side_key}_hold_time_ns"][idx_this]
            filled_mask_q, q_arrival, avg_q = _queue_position_model(
                "passive_at_touch",
                cancel_eval_window=50,
                label_filled=filled_lbl,
                label_exit_reason=exit_reason_lbl,
                label_hold_time_ns=hold_time_lbl,
            )
            edge = _entry_price_edge_ticks("passive_at_touch",
                                           ES_SPREAD_TICKS_RTH_DEFAULT)
            for k_local, ti_local in enumerate(np.where(mask_this)[0]):
                ti = passive_idx_in_trades[ti_local]
                eidx = idx_this[k_local]
                if filled_mask_q[k_local] and mask_30s[eidx]:
                    lr = float(tgt_lr_30s[eidx]) * PRICE_UNIT_TO_TICKS
                    net_per_trade[ti] = sgn * lr + edge - RT_COMM
                    edge_offsets[ti] = edge
                    filled_per_trade[ti] = True
                    adv_sel_30s_per_trade[ti] = min(0.0, sgn * lr)

    n_passive = len(passive_idx_in_trades)
    n_market = n_trades - n_passive
    n_filled_canon = int(filled_per_trade.sum())
    print(f"[canon] trades={n_trades} (market={n_market}, passive={n_passive})")
    print(f"[canon] canonical_filled={n_filled_canon} / {n_trades}")
    diag = {
        "n_trades_input": n_trades,
        "n_market": n_market,
        "n_passive": n_passive,
        "n_canonical_filled": n_filled_canon,
        "fill_rate_canon": n_filled_canon / max(1, n_trades),
        "adv_sel_30s_avg": float(np.nanmean(adv_sel_30s_per_trade))
            if np.isfinite(np.nanmean(adv_sel_30s_per_trade)) else 0.0,
    }
    return net_per_trade, filled_per_trade, ts_per_trade, diag


# -----------------------------------------------------------------------------
# Canonical rules baseline (IDENTICAL to v1)
# -----------------------------------------------------------------------------
def canonical_rules_baseline(npz_path, labels_dir, holdout_frac):
    config = TradeConfig(
        side="short", horizon="5s", confidence_threshold=0.005,
        order_type="passive_at_touch", cancel_eval_window=40, hold_seconds=10.0,
    )
    d = np.load(npz_path, allow_pickle=True)
    dates_all = [str(x) for x in d["oot_dates"]]
    n_hold = max(1, int(round(len(dates_all) * holdout_frac)))
    held_dates = dates_all[-n_hold:]
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
    row = summarize(net, ts_filled, label="rules_j6_top005_short_passive_hold10s_CANON")
    row["adv_sel_30s_avg"] = float(ledger.adverse_selection_cost_ticks_avg)
    return row


# -----------------------------------------------------------------------------
# v1 canonical row (for context, read from existing v1 CSV)
# -----------------------------------------------------------------------------
def v1_canon_row():
    if not LEGACY_V1_CSV.exists():
        return {"name": "ppo_v1_canonical_REF_missing", "n_trades": 0,
                "fills": 0, "ticks_per_trade": float("nan"),
                "ticks_total": float("nan"), "sharpe_sqrtN": float("nan"),
                "sortino_sqrtN": float("nan"), "profit_factor": float("nan"),
                "win_rate": float("nan"), "max_dc_ticks": float("nan"),
                "adv_sel_30s_avg": float("nan"), "day_conc": float("nan"),
                "pass_hc344": False}
    df = pd.read_csv(LEGACY_V1_CSV)
    # Prefer the all-attempts row for headline comparison
    cand = df[df["name"] == "ppo_v3_3_canonical_replay"]
    if cand.empty:
        cand = df.head(1)
    r = cand.iloc[0].to_dict()
    r["name"] = "ppo_v1_canonical_REF"
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
        # Action distribution
        total = float(max(1, action_counts.sum()))
        names = ["hold", "bid", "ask", "mkt_buy", "mkt_sell", "cancel"]
        for nm, c in zip(names, action_counts):
            mlflow.log_metric(f"action_{nm}_count", float(c))
            mlflow.log_metric(f"action_{nm}_pct", 100.0 * float(c) / total)
        for r in rows:
            prefix = (
                "ppo_v2_canon" if r["name"].startswith("ppo_v3_3_v2")
                else "rules_canon" if r["name"].startswith("rules_")
                else "ppo_v1_ref" if r["name"].startswith("ppo_v1")
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
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mlflow-uri", default="http://localhost:5000")
    ap.add_argument("--mlflow-experiment",
                    default="RL_v3_3_smart_exec_v2_sized_reward")
    args = ap.parse_args()

    print(f"[main] start @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[main] args = {vars(args)}")

    trades, N, action_counts = rollout_ppo(
        model_path=Path(args.model), npz_path=Path(args.npz),
        n_episodes=args.n_episodes, holdout_frac=args.holdout_frac, seed=args.seed,
    )
    if not trades:
        print("[ERROR] PPO v2 produced ZERO trades. Cannot canonical-reprice.")
        # Still write action distribution for diagnostics
        total = float(max(1, action_counts.sum()))
        print(f"[diag] action_pct: HOLD={100*action_counts[0]/total:.2f} "
              f"BID={100*action_counts[1]/total:.2f} "
              f"ASK={100*action_counts[2]/total:.2f} "
              f"MKT_BUY={100*action_counts[3]/total:.2f} "
              f"MKT_SELL={100*action_counts[4]/total:.2f} "
              f"CANCEL={100*action_counts[5]/total:.2f}")
        sys.exit(2)

    raw_ledger_csv = OUT_DIR / "ppo_v2_1_canonical_replay_raw_ledger.csv"
    pd.DataFrame(trades).to_csv(raw_ledger_csv, index=False)
    print(f"[main] wrote raw ledger: {raw_ledger_csv}")

    net_canon, filled_canon, ts_canon, canon_diag = canonical_reprice(
        trades=trades, npz_path=Path(args.npz), labels_dir=Path(args.labels_dir),
    )
    valid_mask = ts_canon >= 0
    net_for_summary = net_canon[valid_mask]
    ts_for_summary = ts_canon[valid_mask]
    row_canon = summarize(net_for_summary, ts_for_summary,
                          label="ppo_v3_3_v2_canonical_replay")
    row_canon["adv_sel_30s_avg"] = canon_diag["adv_sel_30s_avg"]
    row_canon["fills"] = int(filled_canon.sum())
    print(f"[main] canon PPO v2: {row_canon}")

    fills_only_net = net_canon[filled_canon]
    fills_only_ts = ts_canon[filled_canon]
    row_canon_fills = summarize(fills_only_net, fills_only_ts,
                                 label="ppo_v3_3_v2_canonical_replay_fills_only")
    row_canon_fills["adv_sel_30s_avg"] = canon_diag["adv_sel_30s_avg"]
    print(f"[main] canon PPO v2 fills-only: {row_canon_fills}")

    row_rules = canonical_rules_baseline(
        npz_path=Path(args.npz), labels_dir=Path(args.labels_dir),
        holdout_frac=args.holdout_frac,
    )
    print(f"[main] canon rules: {row_rules}")

    row_v1_ref = v1_canon_row()
    print(f"[main] v1 ref: {row_v1_ref}")

    # Action distribution row
    total = float(max(1, action_counts.sum()))
    action_row = {
        "name": "ppo_v2_action_distribution_PCT",
        "n_trades": int(action_counts.sum()),
        "fills": 0,
        "ticks_per_trade": 100.0 * action_counts[0] / total,   # HOLD%
        "ticks_total": 100.0 * action_counts[1] / total,       # BID%
        "sharpe_sqrtN": 100.0 * action_counts[2] / total,      # ASK%
        "sortino_sqrtN": 100.0 * action_counts[3] / total,     # MKT_BUY%
        "profit_factor": 100.0 * action_counts[4] / total,     # MKT_SELL%
        "win_rate": 100.0 * action_counts[5] / total,          # CANCEL%
        "max_dc_ticks": float("nan"),
        "adv_sel_30s_avg": float("nan"),
        "day_conc": float("nan"),
        "pass_hc344": False,
    }

    fieldnames = [
        "name", "n_trades", "fills", "ticks_per_trade", "ticks_total",
        "sharpe_sqrtN", "sortino_sqrtN", "profit_factor", "win_rate",
        "max_dc_ticks", "adv_sel_30s_avg", "day_conc", "pass_hc344",
    ]
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in (row_canon, row_canon_fills, row_rules, row_v1_ref, action_row):
            for k in fieldnames:
                r.setdefault(k, float("nan"))
            w.writerow(r)
    print(f"[main] wrote CSV: {args.output_csv}")

    rid = log_mlflow(
        uri=args.mlflow_uri, experiment=args.mlflow_experiment,
        run_name="ppo_v3_3_v2_sized_canonical_eval",
        rows=[row_canon, row_canon_fills, row_rules, row_v1_ref],
        args_dict=vars(args), action_counts=action_counts,
    )

    print("\n========== HC #397 CANONICAL COMPARISON — PPO v2 ==========")
    print(f"{'name':<55} {'tk/tr':>8} {'fills':>6} {'Shrp':>8} {'PF':>6} {'WR%':>6} {'dayC':>6} {'pass':>5}")
    for r in (row_canon, row_canon_fills, row_rules, row_v1_ref):
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
              f"{f('day_conc','.3f'):>6} {str(r.get('pass_hc344','?')):>5}")
    names = ["HOLD", "BID", "ASK", "MKT_BUY", "MKT_SELL", "CANCEL"]
    print("\n========== ACTION DISTRIBUTION (PPO v2) ==========")
    for nm, c in zip(names, action_counts):
        print(f"  {nm:<10} {int(c):>10}  ({100.0*c/total:.2f}%)")
    print(f"\nmlflow_run_id: {rid}")
    print(f"[main] DONE @ {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
