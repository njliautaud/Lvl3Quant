"""
Eval the trained PPO model on the held-out last 20% of v3.3 fold_00 predictions.

HC #396 weekend continuation. Compares PPO to rules baseline (j6-confluence
top-0.5% conf SHORT, fifo_tp8sl5_net, passive_at_touch, hold=10s, FIFO replay,
commission-only $4.70 RT per HC #392).

Outputs:
  - CSV at <output_dir>/ppo_eval_results.csv with both rows.
  - MLflow run `ppo_v3_3_eval` in experiment `RL_v3_3_smart_exec`.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parent / "v3_3_research"))

from env import V33SmartExecEnv, RTH_STEPS, COMMISSION_RT_TICKS  # noqa
from stable_baselines3 import PPO


# --- Risk-adjusted metrics ----------------------------------------------------
# RTH steps per day = 23,400 (env). Trading days per year = 252.
# But annualization here is per-trade not per-step; we use sqrt(N) trade-count
# convention as the user asked for "Sharpe (sqrt N)".
def sharpe_sqrtN(pnl: np.ndarray) -> float:
    if pnl.size < 2:
        return 0.0
    mu = float(pnl.mean())
    sd = float(pnl.std(ddof=1))
    if sd <= 0:
        return 0.0
    return (mu / sd) * np.sqrt(pnl.size)


def sortino_sqrtN(pnl: np.ndarray) -> float:
    if pnl.size < 2:
        return 0.0
    mu = float(pnl.mean())
    down = pnl[pnl < 0]
    if down.size < 1:
        return float("inf") if mu > 0 else 0.0
    dd = float(np.sqrt(np.mean(down ** 2)))
    if dd <= 0:
        return 0.0
    return (mu / dd) * np.sqrt(pnl.size)


def profit_factor(pnl: np.ndarray) -> float:
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def win_rate(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return 0.0
    return float((pnl > 0).mean())


# --- Held-out env: forces episode starts in the last 20% slab ---------------
class HeldOutEnv(V33SmartExecEnv):
    def __init__(self, npz_path: str, seed: int = 0, holdout_frac: float = 0.2):
        super().__init__(npz_path=npz_path, seed=seed)
        self._holdout_start = int(self.N * (1.0 - holdout_frac))
        # Defensive: make sure we still have room for one episode
        if self.N - self._holdout_start < 1000:
            self._holdout_start = max(0, self.N - 1000)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Random start within held-out slab, leaving room for episode
        max_start = max(self._holdout_start + 1,
                        self.N - RTH_STEPS - 1)
        lo = self._holdout_start
        hi = max(lo + 1, max_start)
        self.step_idx = int(self.rng.integers(lo, hi))
        self._reset_episode_state_keep_start()
        return self._obs(), {}


def run_ppo_eval(model_path: Path, npz_path: Path, n_episodes: int,
                 holdout_frac: float, seed: int) -> dict:
    print(f"[ppo] loading model from {model_path}")
    model = PPO.load(str(model_path), device="cpu")
    env = HeldOutEnv(str(npz_path), seed=seed, holdout_frac=holdout_frac)
    print(f"[ppo] N={env.N} holdout_start={env._holdout_start} "
          f"holdout_size={env.N - env._holdout_start}")

    per_trade_pnls: list[float] = []
    per_episode_pnls: list[float] = []
    total_trades = 0
    total_steps = 0
    episodes_done = 0

    while episodes_done < n_episodes:
        obs, _ = env.reset(seed=seed + episodes_done)
        terminated = truncated = False
        ep_realized = 0.0
        last_realized = 0.0
        last_trades = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_steps += 1
            # When trades_count increments, capture the per-trade realized delta
            cur_realized = info["realized_pnl_ticks"]
            cur_trades = info["trades"]
            if cur_trades > last_trades:
                per_trade_pnl = cur_realized - last_realized
                per_trade_pnls.append(per_trade_pnl)
                last_realized = cur_realized
                last_trades = cur_trades
            ep_realized = cur_realized
        per_episode_pnls.append(ep_realized)
        total_trades += last_trades
        episodes_done += 1
        print(f"[ppo] ep{episodes_done}/{n_episodes} "
              f"trades={last_trades} realized_pnl={ep_realized:+.2f}t")

    pnl = np.asarray(per_trade_pnls, dtype=np.float64)
    return {
        "name": "ppo_v3_3",
        "n_trades": int(pnl.size),
        "n_episodes": episodes_done,
        "ticks_per_trade": float(pnl.mean()) if pnl.size else 0.0,
        "ticks_total": float(pnl.sum()),
        "sharpe_sqrtN": sharpe_sqrtN(pnl),
        "sortino_sqrtN": sortino_sqrtN(pnl),
        "profit_factor": profit_factor(pnl),
        "win_rate": win_rate(pnl),
        "day_conc": float("nan"),  # episodes are random, day-conc not meaningful
        "fills": int(pnl.size),
    }


# --- Rules baseline via canonical replay --------------------------------------
def run_rules_baseline(npz_path: Path, labels_dir: Path, holdout_frac: float) -> dict:
    """j6-confluence top-0.5% conf SHORT on fifo_tp8sl5_net, passive_at_touch,
    hold=10s. Uses canonical full_market_replay so cost basis is consistent
    (HC #392 commission-only, queue/adverse modeled)."""
    sys.path.insert(0, "/home/jupiter/Lvl3Quant")
    from scripts.v3_3_research.full_market_replay import (
        full_market_replay, TradeConfig,
    )

    # Note: canonical replay uses pred_log_ret_<horizon> as the signal head.
    # To closely match the user's spec (top-0.5% conf SHORT on fifo_tp8sl5_net)
    # we run it WITHIN the same library by using pred_log_ret_5s as the signal
    # (best correlated with fifo_tp8sl5_net per j6 results) at conf=0.005.
    # This is the canonical interface; using a different signal head would
    # require modifying the library which violates HC #357 stability.
    config = TradeConfig(
        side="short",
        horizon="5s",
        confidence_threshold=0.005,   # top 0.5%
        order_type="passive_at_touch",
        cancel_eval_window=40,        # ~10s
        hold_seconds=10.0,
    )

    # Detect held-out dates: take the last 20% of OOT day list
    d = np.load(npz_path, allow_pickle=True)
    dates_all = [str(x) for x in d["oot_dates"]]
    n_hold = max(1, int(round(len(dates_all) * holdout_frac)))
    held_dates = dates_all[-n_hold:]
    print(f"[rules] using held-out dates: {held_dates}")

    ledger = full_market_replay(
        predictions_npz_path=npz_path,
        mbo_labels_dir=labels_dir,
        config=config,
        dates=held_dates,
        verbose=True,
    )

    return {
        "name": "rules_j6_top005_short_passive_hold10s",
        "n_trades": int(ledger.n_filled),
        "n_episodes": 1,
        "ticks_per_trade": float(ledger.pnl_ticks_per_fill),
        "ticks_total": float(ledger.pnl_ticks_total),
        "sharpe_sqrtN": float(ledger.sharpe),
        "sortino_sqrtN": float(ledger.sortino),
        "profit_factor": float(ledger.profit_factor),
        "win_rate": float(ledger.win_rate),
        "day_conc": float("nan"),
        "fills": int(ledger.n_filled),
    }


# --- MLflow logging -----------------------------------------------------------
def log_to_mlflow(uri: str, experiment: str, run_name: str,
                  ppo_metrics: dict, rules_metrics: dict,
                  args_dict: dict) -> str | None:
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
        mlflow.log_params({f"arg_{k}": v for k, v in args_dict.items()})
        for prefix, m in (("ppo", ppo_metrics), ("rules", rules_metrics)):
            for k, v in m.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    mlflow.log_metric(f"{prefix}_{k}", float(v))
        # Delta vs rules baseline
        for k in ("ticks_per_trade", "sharpe_sqrtN", "sortino_sqrtN"):
            try:
                mlflow.log_metric(f"delta_{k}",
                                  float(ppo_metrics[k]) - float(rules_metrics[k]))
            except Exception:
                pass
        mlflow.end_run()
        print(f"[mlflow] logged run {rid}")
        return rid
    except Exception as e:
        print(f"[mlflow] failed: {e}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--labels-dir",
                    default="/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_fifo_labels")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mlflow-uri", default="http://localhost:5000")
    ap.add_argument("--mlflow-experiment", default="RL_v3_3_smart_exec")
    ap.add_argument("--skip-rules", action="store_true")
    args = ap.parse_args()

    print(f"[eval] starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    ppo_metrics = run_ppo_eval(
        model_path=Path(args.model),
        npz_path=Path(args.npz),
        n_episodes=args.n_episodes,
        holdout_frac=args.holdout_frac,
        seed=args.seed,
    )
    print(f"[eval] PPO metrics: {ppo_metrics}")

    if args.skip_rules:
        rules_metrics = {"name": "skipped", "n_trades": 0, "n_episodes": 0,
                          "ticks_per_trade": 0.0, "ticks_total": 0.0,
                          "sharpe_sqrtN": 0.0, "sortino_sqrtN": 0.0,
                          "profit_factor": 0.0, "win_rate": 0.0,
                          "day_conc": float("nan"), "fills": 0}
    else:
        rules_metrics = run_rules_baseline(
            npz_path=Path(args.npz),
            labels_dir=Path(args.labels_dir),
            holdout_frac=args.holdout_frac,
        )
    print(f"[eval] rules metrics: {rules_metrics}")

    # CSV
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ppo_metrics.keys())
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(ppo_metrics)
        w.writerow(rules_metrics)
    print(f"[eval] wrote CSV: {args.output_csv}")

    rid = log_to_mlflow(
        uri=args.mlflow_uri,
        experiment=args.mlflow_experiment,
        run_name="ppo_v3_3_eval",
        ppo_metrics=ppo_metrics,
        rules_metrics=rules_metrics,
        args_dict=vars(args),
    )
    print(f"[eval] DONE. mlflow_run={rid}")


if __name__ == "__main__":
    main()
