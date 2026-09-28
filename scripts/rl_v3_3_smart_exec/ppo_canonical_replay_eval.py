#!/usr/bin/env python3
"""
HC #397 canonical-replay re-eval of trained PPO v1 policy.

Background:
  The v1 PPO eval (ppo_eval_results.csv) reported +1.04 ticks/trade headline.
  That number used the env's INTERNAL reward, which is only PARTIALLY canonical:
    - Passive limit fills use `target_fifo_tp8sl5_net` (CANONICAL 5-comp label)
    - Market orders use a toy `target_log_ret_30s - 1.376 ticks` formula
      (NOT canonical; also violates HC #392 — market should be commission-only
       0.376 ticks because spread is already in canonical fill prices)
    - Hold/cancel MTM uses `target_log_ret_1s` (research proxy)

  This script re-runs the trained PPO policy deterministically over the SAME
  held-out slab the original eval used (last 20% of fold_00_predictions, 20
  episodes, seed=42), captures its trade ledger via a thin instrumentation
  Gym wrapper (we do NOT modify env.py per HC #357 stability rule), then for
  each trade re-prices PnL through the canonical full_market_replay library
  primitives with HC #392 commission-only basis.

Outputs:
  - CSV with 3 rows: canonical PPO, canonical rules, PPO env-reward (legacy)
  - MLflow run `ppo_v3_3_canonical_eval` in experiment `RL_v3_3_smart_exec`

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

from env import (  # noqa: E402
    V33SmartExecEnv,
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
MODEL_PATH = LVL3 / "output/rl_v3_3_smart_exec/ppo_v3_3_final.zip"
OUT_DIR = LVL3 / "output/rl_v3_3_smart_exec"
OUT_CSV = OUT_DIR / "ppo_canonical_replay.csv"
LEGACY_CSV = OUT_DIR / "ppo_eval_results.csv"

RT_COMM = ES_RT_COMMISSION_TICKS_DEFAULT  # 0.376
DAY_CONC_GATE = 0.20


# -----------------------------------------------------------------------------
# Held-out env (mirrors eval_ppo.HeldOutEnv exactly)
# -----------------------------------------------------------------------------
class HeldOutEnv(V33SmartExecEnv):
    def __init__(self, npz_path, seed=0, holdout_frac=0.2):
        super().__init__(npz_path=npz_path, seed=seed)
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
# Trade-ledger instrumentation wrapper (NOT a modification of env.py)
# -----------------------------------------------------------------------------
class TradeLogWrapper(gym.Wrapper):
    """Inspects env state pre- and post-step to detect:
       - Position OPEN: position_side 0 -> +/-1.  Record entry_idx (step_idx
         AT the moment the position was opened), side, and action_type:
           passive       (label-based open from pending_order resolution)
           market        (immediate open from A_MKT_BUY / A_MKT_SELL)
       - Position CLOSE: position_side +/-1 -> 0.  Record exit_idx and the
         env's reward attributable to the close.

    Stores ledger entries in self.trades.  Each trade is one dict:
       episode, entry_idx, exit_idx, side {+1,-1}, action_type {passive,market},
       env_realized_at_close (ticks), env_reward_at_close (ticks).

    Does NOT alter env behavior — pure observation.
    """

    def __init__(self, env: HeldOutEnv):
        super().__init__(env)
        self.trades: list[dict] = []
        self._episode = -1
        self._open_trade: dict | None = None
        self._last_action: int = -1

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._episode += 1
        self._open_trade = None
        self._last_action = -1
        return obs, info

    def step(self, action: int):
        e = self.env
        action_int = int(action)
        prev_side = int(e.position_side)
        prev_pending = int(e.pending_order)

        obs, reward, terminated, truncated, info = self.env.step(action_int)

        new_side = int(e.position_side)

        # OPEN detection: prev_side == 0 and new_side != 0
        if prev_side == 0 and new_side != 0:
            # Determine action_type:
            #   - If passive pending resolved THIS step at the env.step's pending
            #     resolution block, the entry happens at "i" = pre-step step_idx,
            #     which equals the current e.step_idx - 1 after step.
            #   - If market open, entry also at pre-step step_idx == e.step_idx - 1.
            entry_idx_global = int(e.entry_idx)  # env sets this to "i"
            if action_int in (A_MKT_BUY, A_MKT_SELL):
                atype = "market"
            elif prev_pending != 0:
                # Passive fill resolved at this step (pending was non-zero before
                # the step). Even if action == hold/cancel, the pre-step pending
                # resolution block could fill us.
                atype = "passive"
            else:
                # Edge case: shouldn't normally happen. Default to passive.
                atype = "passive"

            self._open_trade = {
                "episode": self._episode,
                "entry_idx": entry_idx_global,
                "side": new_side,
                "action_type": atype,
                "open_step": int(e.step_idx),
            }

        # CLOSE detection: prev_side != 0 and new_side == 0
        if prev_side != 0 and new_side == 0 and self._open_trade is not None:
            self._open_trade["exit_idx"] = int(e.step_idx) - 1  # exit at "i"
            self._open_trade["env_realized_at_close"] = float(e.realized_pnl_ticks)
            self._open_trade["env_step_reward_at_close"] = float(reward)
            self.trades.append(self._open_trade)
            self._open_trade = None

        self._last_action = action_int
        return obs, reward, terminated, truncated, info


# -----------------------------------------------------------------------------
# Risk-adjusted metrics (matches summarize() in v33_sized_replay.py)
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
    # Sharpe using sqrt(N) convention (per HC #397 / eval_ppo.py)
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
        "name": label,
        "n_trades": n_filled,
        "ticks_per_trade": total / max(1, n_filled),
        "ticks_total": total,
        "sharpe_sqrtN": sharpe,
        "sortino_sqrtN": sortino,
        "profit_factor": pf,
        "win_rate": wr,
        "max_dc_ticks": max_dc,
        "day_conc": day_conc,
        "pass_hc344": pass_hc344,
        "fills": n_filled,
    }


# -----------------------------------------------------------------------------
# PPO rollout + ledger extraction
# -----------------------------------------------------------------------------
def rollout_ppo(model_path: Path, npz_path: Path, n_episodes: int,
                holdout_frac: float, seed: int) -> tuple[list[dict], int]:
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
    return env.trades, base_env.N


# -----------------------------------------------------------------------------
# Canonical re-price of a PPO trade ledger
# -----------------------------------------------------------------------------
def canonical_reprice(
    trades: list[dict],
    npz_path: Path,
    labels_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """For each PPO trade, compute canonical-replay PnL in ticks.

    Logic:
      - Load OOT dates, FIFO labels, per-horizon target log-rets.
      - For each trade i with entry_idx, side, action_type:
          * If action_type == 'market':
              - Fill is GUARANTEED (book crossed). HC #392: cost = commission RT
                only (0.376 ticks); NO extra spread crossing — the canonical
                replay primitive `_entry_price_edge_ticks('ioc_market', ...)`
                returns -spread_ticks_rth which would DOUBLE-count vs HC #392.
                So we manually set edge_offset = 0 for market under HC #392.
              - Exit: use target_log_ret_30s at entry_idx (the env's nominal
                hold horizon; FORCED_EXIT at MAX_HOLD_STEPS=240 => 60s; we use
                30s exit-MTM as the closest canonical horizon in the replay
                library since exit happens after a position-age step loop).
              - Net = side * tgt_lr_30s[entry_idx] + edge_offset - rt_comm
          * If action_type == 'passive':
              - Apply queue-position model deflator using the FIFO label fields
                at entry_idx: tp4sl3_{side}_filled, exit_reason, hold_time_ns.
              - cancel_window = 50 (env's PASSIVE_FILL_WINDOW).
              - If queue model says NOT filled => trade contributes 0 (cancelled).
              - If filled: edge_offset = 0 (passive at touch), exit horizon =
                30s (env hold cap), net = side * tgt_lr_30s[entry_idx] - rt_comm.

    Returns:
      net (n_trades,) — canonical net PnL per trade (0 for canonical cancels)
      filled_mask (n_trades,) — True where canonical replay says filled
      diag — dict of diagnostics
    """
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

    # Buckets for queue-model batch call: passive trades only
    passive_idx_in_trades = []
    passive_entry_idx = []
    passive_side = []

    for ti, tr in enumerate(trades):
        entry_idx = int(tr["entry_idx"])
        side = int(tr["side"])
        if entry_idx < 0 or entry_idx >= n_total:
            # Out of canonical data range — skip
            continue
        ts_per_trade[ti] = int(fifo["ts_ns"][entry_idx])
        if tr["action_type"] == "market":
            # GUARANTEED fill — HC #392: commission-only basis
            if mask_30s[entry_idx]:
                lr = float(tgt_lr_30s[entry_idx]) * PRICE_UNIT_TO_TICKS
                # HC #392: NO extra spread cross — entry already crossed; canonical
                # replay library counts spread as -1 tick edge_offset for ioc_market,
                # but HC #392 says that's already in the fill price (book ask/bid)
                # in real 5-comp replay. So we use commission-only.
                edge = 0.0
                net_per_trade[ti] = side * lr + edge - RT_COMM
                edge_offsets[ti] = edge
                filled_per_trade[ti] = True
                adv_sel_30s_per_trade[ti] = min(0.0, side * lr)
            else:
                # No mask — treat as cancelled
                pass
        elif tr["action_type"] == "passive":
            passive_idx_in_trades.append(ti)
            passive_entry_idx.append(entry_idx)
            passive_side.append(side)
        else:
            pass  # unknown — skip

    # Batch-apply queue model to passive trades
    if passive_idx_in_trades:
        p_idx = np.asarray(passive_entry_idx, dtype=np.int64)
        p_side = np.asarray(passive_side, dtype=np.int32)

        # Pull label arrays per side. Mixed-side handling: do longs then shorts.
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
                cancel_eval_window=50,  # env's PASSIVE_FILL_WINDOW
                label_filled=filled_lbl,
                label_exit_reason=exit_reason_lbl,
                label_hold_time_ns=hold_time_lbl,
            )

            # Compute net for filled passive trades using 30s exit horizon
            edge = _entry_price_edge_ticks("passive_at_touch",
                                           ES_SPREAD_TICKS_RTH_DEFAULT)  # = 0
            for k_local, ti_local in enumerate(np.where(mask_this)[0]):
                ti = passive_idx_in_trades[ti_local]
                eidx = idx_this[k_local]
                if filled_mask_q[k_local] and mask_30s[eidx]:
                    lr = float(tgt_lr_30s[eidx]) * PRICE_UNIT_TO_TICKS
                    net_per_trade[ti] = sgn * lr + edge - RT_COMM
                    edge_offsets[ti] = edge
                    filled_per_trade[ti] = True
                    adv_sel_30s_per_trade[ti] = min(0.0, sgn * lr)
                else:
                    # canonical replay says cancelled — 0 PnL, no commission
                    pass

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
# Canonical rules baseline (mirrors eval_ppo.run_rules_baseline)
# -----------------------------------------------------------------------------
def canonical_rules_baseline(npz_path: Path, labels_dir: Path,
                             holdout_frac: float) -> dict:
    """j6-confluence top-0.5% conf SHORT on log_ret_5s, passive_at_touch,
    cancel_window=40, hold=10s. Same as eval_ppo.py rules baseline."""
    config = TradeConfig(
        side="short",
        horizon="5s",
        confidence_threshold=0.005,
        order_type="passive_at_touch",
        cancel_eval_window=40,
        hold_seconds=10.0,
    )
    d = np.load(npz_path, allow_pickle=True)
    dates_all = [str(x) for x in d["oot_dates"]]
    n_hold = max(1, int(round(len(dates_all) * holdout_frac)))
    held_dates = dates_all[-n_hold:]
    print(f"[rules] held_dates={held_dates}")

    ledger = full_market_replay(
        predictions_npz_path=npz_path,
        mbo_labels_dir=labels_dir,
        config=config,
        dates=held_dates,
        verbose=True,
    )
    # Convert to Sharpe(sqrt N) for apples-to-apples with PPO summary
    net = np.asarray(
        ledger.per_trade_df.loc[ledger.per_trade_df["filled"], "net_ticks"].values,
        dtype=np.float64,
    )
    net = net[np.isfinite(net)]
    ts_filled = ledger.per_trade_df.loc[ledger.per_trade_df["filled"], "timestamp"].values
    ts_filled = np.asarray(ts_filled, dtype=np.int64)
    row = summarize(net, ts_filled, label="rules_j6_top005_short_passive_hold10s_CANON")
    # Replace adv_sel_30s_avg with library's value where available
    row["adv_sel_30s_avg"] = float(ledger.adverse_selection_cost_ticks_avg)
    return row


# -----------------------------------------------------------------------------
# Legacy env-reward row (re-read from existing CSV for context per HC #397)
# -----------------------------------------------------------------------------
def legacy_row() -> dict:
    if not LEGACY_CSV.exists():
        return {"name": "ppo_env_reward_LEGACY", "n_trades": 0, "ticks_per_trade": 0.0,
                "ticks_total": 0.0, "sharpe_sqrtN": 0.0, "sortino_sqrtN": 0.0,
                "profit_factor": 0.0, "win_rate": 0.0, "max_dc_ticks": float("nan"),
                "adv_sel_30s_avg": float("nan"),
                "day_conc": float("nan"), "pass_hc344": False, "fills": 0}
    df = pd.read_csv(LEGACY_CSV)
    r = df[df["name"] == "ppo_v3_3"].iloc[0]
    return {
        "name": "ppo_env_reward_LEGACY_for_context",
        "n_trades": int(r["n_trades"]),
        "ticks_per_trade": float(r["ticks_per_trade"]),
        "ticks_total": float(r["ticks_total"]),
        "sharpe_sqrtN": float(r["sharpe_sqrtN"]),
        "sortino_sqrtN": float(r["sortino_sqrtN"]),
        "profit_factor": float(r["profit_factor"]),
        "win_rate": float(r["win_rate"]) * 100.0,  # legacy stored as fraction
        "max_dc_ticks": float("nan"),
        "adv_sel_30s_avg": float("nan"),
        "day_conc": float("nan"),
        "pass_hc344": False,
        "fills": int(r["fills"]),
    }


# -----------------------------------------------------------------------------
# MLflow
# -----------------------------------------------------------------------------
def log_mlflow(uri: str, experiment: str, run_name: str,
               rows: list[dict], args_dict: dict) -> str | None:
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
        for r in rows:
            prefix = (
                "ppo_canon" if r["name"].startswith("ppo_v3_3_canonical")
                else "rules_canon" if r["name"].startswith("rules_")
                else "ppo_legacy"
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
    ap.add_argument("--mlflow-experiment", default="RL_v3_3_smart_exec")
    args = ap.parse_args()

    print(f"[main] start @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[main] args = {vars(args)}")

    # 1) PPO rollout → trade ledger
    trades, N = rollout_ppo(
        model_path=Path(args.model), npz_path=Path(args.npz),
        n_episodes=args.n_episodes, holdout_frac=args.holdout_frac, seed=args.seed,
    )

    if not trades:
        print("[ERROR] PPO produced ZERO trades. Cannot canonical-reprice.")
        sys.exit(2)

    # Persist raw ledger
    raw_ledger_csv = OUT_DIR / "ppo_canonical_replay_raw_ledger.csv"
    pd.DataFrame(trades).to_csv(raw_ledger_csv, index=False)
    print(f"[main] wrote raw ledger: {raw_ledger_csv}")

    # 2) Canonical re-price
    net_canon, filled_canon, ts_canon, canon_diag = canonical_reprice(
        trades=trades, npz_path=Path(args.npz), labels_dir=Path(args.labels_dir),
    )
    # For headline metrics: include ALL trades that PPO attempted (cancelled
    # passive trades contribute 0). Per HC #344 day_conc should be computed
    # on attempted-day basis using ts of each trade.
    valid_mask = ts_canon >= 0
    net_for_summary = net_canon[valid_mask]
    ts_for_summary = ts_canon[valid_mask]
    row_canon = summarize(net_for_summary, ts_for_summary,
                          label="ppo_v3_3_canonical_replay")
    row_canon["adv_sel_30s_avg"] = canon_diag["adv_sel_30s_avg"]
    row_canon["fills"] = int(filled_canon.sum())
    print(f"[main] canon PPO: {row_canon}")

    # 2b) Also a "fills-only" view (only canonical-filled trades) for ticks/fill
    fills_only_net = net_canon[filled_canon]
    fills_only_ts = ts_canon[filled_canon]
    row_canon_fills = summarize(fills_only_net, fills_only_ts,
                                 label="ppo_v3_3_canonical_replay_fills_only")
    row_canon_fills["adv_sel_30s_avg"] = canon_diag["adv_sel_30s_avg"]
    print(f"[main] canon PPO fills-only: {row_canon_fills}")

    # 3) Canonical rules baseline
    row_rules = canonical_rules_baseline(
        npz_path=Path(args.npz), labels_dir=Path(args.labels_dir),
        holdout_frac=args.holdout_frac,
    )
    print(f"[main] canon rules: {row_rules}")

    # 4) Legacy env-reward row for context
    row_legacy = legacy_row()
    print(f"[main] legacy: {row_legacy}")

    # 5) Write CSV — all rows
    fieldnames = [
        "name", "n_trades", "fills", "ticks_per_trade", "ticks_total",
        "sharpe_sqrtN", "sortino_sqrtN", "profit_factor", "win_rate",
        "max_dc_ticks", "adv_sel_30s_avg", "day_conc", "pass_hc344",
    ]
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in (row_canon, row_canon_fills, row_rules, row_legacy):
            # Ensure all keys exist; fill missing with NaN
            for k in fieldnames:
                r.setdefault(k, float("nan"))
            w.writerow(r)
    print(f"[main] wrote CSV: {args.output_csv}")

    # 6) MLflow
    rid = log_mlflow(
        uri=args.mlflow_uri, experiment=args.mlflow_experiment,
        run_name="ppo_v3_3_canonical_eval",
        rows=[row_canon, row_canon_fills, row_rules, row_legacy],
        args_dict=vars(args),
    )

    # 7) Final comparison printout (HC #397 format)
    print("\n========== HC #397 CANONICAL COMPARISON ==========")
    print(f"{'name':<55} {'tk/tr':>8} {'fills':>6} {'Shrp':>8} {'PF':>6} {'WR%':>6} {'dayC':>6} {'pass':>5}")
    for r in (row_canon, row_canon_fills, row_rules, row_legacy):
        def f(k, fmt=".3f"):
            v = r.get(k, float("nan"))
            if isinstance(v, bool):
                return str(v)
            try:
                return f"{float(v):{fmt}}"
            except Exception:
                return "nan"
        print(f"{r['name']:<55} {f('ticks_per_trade'):>8} "
              f"{int(r.get('fills',0)):>6} {f('sharpe_sqrtN','.2f'):>8} "
              f"{f('profit_factor','.2f'):>6} {f('win_rate','.1f'):>6} "
              f"{f('day_conc','.3f'):>6} {str(r.get('pass_hc344','?')):>5}")
    print(f"\nmlflow_run_id: {rid}")
    print(f"[main] DONE @ {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
