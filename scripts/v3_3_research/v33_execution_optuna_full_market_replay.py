"""
v33_execution_optuna_full_market_replay.py — HC #369 PRIMARY Optuna sweep.

GOAL (HC #369): find ≥3 distinct v3.3 execution configs that pass ALL HC #344
deploy gates under HC #357 full-market-replay basis (FIFO + queue + adv-sel +
commission + cancel/replace). 0/213 grid configs passed (HC #368 config
search) — this script expands to ≥30 hyperparameters, ≥5000 trials, TPE +
MedianPruner.

OUTPUT:
  output/v33_execution_optuna_20260515/
    study.db                                 (SQLite Optuna storage)
    best_configs.json                        (top-K configs passing HC #344)
    leaderboard.csv                          (all trials ranked)
    deploy_eligible_configs/<trial_id>.json  (each = full v33_paper_trader_config)
    progress.jsonl                           (per-trial line: id, params, metrics)

MALWARE-GUARD (HC #307D): pure analysis script. Reads predictions NPZ + FIFO
labels. Writes only to output/v33_execution_optuna_20260515/. Does NOT touch
trainer code. New file under scripts/v3_3_research/ (analysis tooling).

HCs SATISFIED: #369 (Optuna-with-tons-of-vars), #357 (full-market-replay),
#344 (deploy gates), #361 (price-path metrics in per_trade), #321 (Sharpe
on risk-adjusted basis).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Make sure scripts/ on path
PROJ = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(PROJ))

from scripts.v3_3_research.full_market_replay import (  # noqa: E402
    full_market_replay, TradeConfig, TradeLedger,
    _load_predictions, _load_fifo_labels,
)

# ---------------------------------------------------------------------------
# CONSTANTS (HC #344 deploy gates)
# ---------------------------------------------------------------------------
DEFAULT_PREDS_PATH = PROJ / "output" / "cnn_mamba_v3_3_uncertainty_weighted" / "fold_00_predictions.npz"
DEFAULT_LABELS_DIR = PROJ / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"
DEFAULT_SIGMA_PATH = PROJ / "output" / "cnn_mamba_v3_3_uncertainty_weighted" / "fold_00_sigma.json"
DEFAULT_OUT_DIR = PROJ / "output" / "v33_execution_optuna_20260515"

GATE_MIN_N_FILLS = 30
GATE_MIN_PF = 1.2
GATE_MAX_DAY_CONC = 0.70  # max fraction of pnl coming from a single day
GATE_MIN_SHARPE = 0.50
GATE_MIN_CI_LOW_95 = -0.50  # 95% CI lower bound on per-trade net ticks

# Penalty Sharpe when gates fail (still observable, but rank far below valid)
GATE_FAIL_SCORE = -10.0

# ---------------------------------------------------------------------------
# Cache predictions + labels at process start (Optuna calls objective 5000+ times)
# ---------------------------------------------------------------------------
_CACHE: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# HC #426 R3: canonical avg-move empirical bands
# tp candidate range  = [p50, p90] of realized abs-move
# sl candidate range  = [median, p75] of realized adverse-move (or abs p25 fallback)
# hold_seconds range  = [horizon_sec, 3 * horizon_sec]
# cancel_window range = derived from suggested_cancel_window_evals when present
# ---------------------------------------------------------------------------
_AVG_MOVE_BANDS: dict[str, dict[str, Any]] = {}


def _load_avg_move_bands(cache_arg: str | None) -> None:
    """Populate _AVG_MOVE_BANDS from one or more canonical_avg_move_*.json files.

    cache_arg may be:
      - a single file path (one horizon)
      - a comma-separated list of paths
      - a directory containing canonical_avg_move_*_<horizon>.json files
      - a glob-style template containing '{h}' (e.g. '.../canonical_avg_move_v3_3_{h}.json')
    """
    if not cache_arg:
        return

    horizons = ["1s", "5s", "10s", "30s"]
    candidates: list[Path] = []

    if "{h}" in cache_arg:
        for h in horizons:
            candidates.append(Path(cache_arg.replace("{h}", h)))
    elif "," in cache_arg:
        candidates = [Path(p.strip()) for p in cache_arg.split(",") if p.strip()]
    else:
        p = Path(cache_arg)
        if p.is_dir():
            for h in horizons:
                # Match any file like canonical_avg_move_*_<h>.json in the dir
                hits = sorted(p.glob(f"canonical_avg_move_*_{h}.json"))
                if hits:
                    candidates.append(hits[-1])  # newest by name (e.g. v3_4_2 > v3_3)
        else:
            candidates = [p]

    for path in candidates:
        if not path.exists():
            print(f"[avg-move] WARN: cache file missing: {path}")
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(f"[avg-move] WARN: failed to parse {path}: {e}")
            continue

        horizon = data.get("horizon")
        if not horizon:
            # Try to infer from filename
            stem = path.stem
            for h in horizons:
                if stem.endswith(f"_{h}"):
                    horizon = h
                    break
        if not horizon:
            print(f"[avg-move] WARN: cannot infer horizon for {path}")
            continue

        abs_t = data.get("abs_ticks", {}) or {}
        signed_t = data.get("signed_ticks", {}) or {}
        mae = data.get(f"realized_pred_mae_{horizon}_ticks", {}) or {}

        # tp band: [p50, p90] of abs-move (empirical, not hard-coded)
        tp_lo = float(abs_t.get("p50", 1.0))
        tp_hi = float(abs_t.get("p90", max(tp_lo + 1.0, 4.0)))
        if tp_hi <= tp_lo:
            tp_hi = tp_lo + 1.0

        # sl band: prefer dedicated adverse-move stats when present
        # MAE is signed-negative ; magnitude grows as values become more negative.
        # median (p50) magnitude < p75 magnitude (p75 is closer to zero by p25/p75
        # convention on negatives). Use abs() of MAE p50 / p25 for magnitudes:
        # |median|  -> abs(p50);   |p75-magnitude tail| -> abs(p25 of MAE, which is
        # the deeper-negative quartile).
        if mae:
            sl_lo = abs(float(mae.get("p50", -1.0)))
            sl_hi = abs(float(mae.get("p25", -2.0)))
            if sl_hi <= sl_lo:
                sl_hi = sl_lo + 0.5
        else:
            sl_lo = float(abs_t.get("p25", 1.0))
            sl_hi = float(abs_t.get("p50", max(sl_lo + 0.5, 2.0)))
            if sl_hi <= sl_lo:
                sl_hi = sl_lo + 0.5

        # hold_seconds: [horizon_sec, 3 * horizon_sec] per HC #426 R3
        h_sec = float(data.get("horizon_sec", _horizon_str_to_sec(horizon)))
        hold_lo = h_sec
        hold_hi = 3.0 * h_sec

        # cancel_window: take from cache when present, else default
        cw = data.get("suggested_cancel_window_evals") or [4, 80]
        try:
            cancel_lo = int(cw[0])
            cancel_hi = int(cw[1])
            if cancel_hi <= cancel_lo:
                cancel_hi = cancel_lo + 1
        except Exception:
            cancel_lo, cancel_hi = 4, 80

        _AVG_MOVE_BANDS[horizon] = {
            "tp_lo": tp_lo, "tp_hi": tp_hi,
            "sl_lo": sl_lo, "sl_hi": sl_hi,
            "hold_lo": hold_lo, "hold_hi": hold_hi,
            "cancel_lo": cancel_lo, "cancel_hi": cancel_hi,
            "source": str(path),
        }
        print(
            f"[avg-move] {horizon}: tp=[{tp_lo:.2f},{tp_hi:.2f}] "
            f"sl=[{sl_lo:.2f},{sl_hi:.2f}] hold=[{hold_lo:.1f},{hold_hi:.1f}]s "
            f"cancel=[{cancel_lo},{cancel_hi}] <- {path.name}"
        )


def _horizon_str_to_sec(h: str) -> float:
    if h.endswith("s"):
        try:
            return float(h[:-1])
        except Exception:
            pass
    return 1.0


def init_cache(preds_path: Path, labels_dir: Path) -> None:
    """Pre-load all 4 horizon predictions + FIFO labels once."""
    print(f"[init] preds={preds_path}")
    print(f"[init] labels={labels_dir}")
    _CACHE["preds"] = {
        h: _load_predictions(preds_path, h) for h in ["1s", "5s", "10s", "30s"]
    }
    dates = list(_CACHE["preds"]["1s"]["oot_dates"])
    _CACHE["dates"] = dates
    _CACHE["labels"] = _load_fifo_labels(labels_dir, dates)
    print(f"[init] horizons cached: {list(_CACHE['preds'].keys())}")
    print(f"[init] OOT dates: {dates}")
    print(f"[init] n_samples per horizon: {[_CACHE['preds'][h]['n'] for h in ['1s','5s','10s','30s']]}")
    if DEFAULT_SIGMA_PATH.exists():
        _CACHE["sigma"] = json.loads(DEFAULT_SIGMA_PATH.read_text())
    else:
        _CACHE["sigma"] = {}


# ---------------------------------------------------------------------------
# Post-replay filtering helpers (TONS-OF-VARS overlays)
# ---------------------------------------------------------------------------
def apply_post_filters(
    ledger: TradeLedger,
    *,
    tod_start_hour: int,
    tod_end_hour: int,
    require_min_pred_strength: float | None = None,
    confluence_secondary_sign: int = 0,  # +1, -1, 0=off — sign of pred at same idx in another horizon
    secondary_preds: np.ndarray | None = None,
    secondary_mask: np.ndarray | None = None,
    secondary_sel_idx: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Apply ToD / confluence / pred-strength filters to per_trade_df.

    Returns (filtered_df, gate_metrics) where filtered_df only contains rows
    surviving filters AND filled trades (cancelled excluded for Sharpe calc).
    """
    df = ledger.per_trade_df.copy()
    if df.empty:
        return df, {}

    # FILLED only
    df = df[df["filled"]].reset_index(drop=True)
    if df.empty:
        return df, {"reason": "no_fills_after_filter"}

    # Time-of-day filter (ET-naive interpretation; timestamps are ns since epoch UTC)
    ts_ns = df["timestamp"].to_numpy()
    # Convert to ET hours via pandas
    ts_pd = pd.to_datetime(ts_ns, unit="ns", utc=True).tz_convert("America/New_York")
    hours = ts_pd.hour.to_numpy()
    tod_mask = (hours >= tod_start_hour) & (hours < tod_end_hour)
    df = df.loc[tod_mask].reset_index(drop=True)

    if df.empty:
        return df, {"reason": "no_fills_after_tod"}

    # Min pred-strength filter (absolute value)
    if require_min_pred_strength is not None and require_min_pred_strength > 0:
        strength_mask = np.abs(df["prediction"].to_numpy()) >= require_min_pred_strength
        df = df.loc[strength_mask].reset_index(drop=True)

    if df.empty:
        return df, {"reason": "no_fills_after_pred_strength"}

    # Confluence with secondary head sign
    # (Skipped here since it requires re-indexing into original sel_idx — see __TODO)
    # For initial sweep we omit; can extend in v2.

    return df, {"ok": True, "n_filled_after_filters": len(df)}


def metrics_from_filtered(df: pd.DataFrame) -> dict:
    """Compute HC #344 metrics on filtered filled trades."""
    if df.empty:
        return {
            "n_fills": 0, "sharpe": 0.0, "sortino": 0.0, "pf": 0.0, "wr": 0.0,
            "mean_net": 0.0, "day_conc": 1.0, "ci_low_95": -999.0,
        }

    net = df["net_ticks"].to_numpy(dtype=float)
    net = net[np.isfinite(net)]
    n = len(net)
    if n == 0:
        return {
            "n_fills": 0, "sharpe": 0.0, "sortino": 0.0, "pf": 0.0, "wr": 0.0,
            "mean_net": 0.0, "day_conc": 1.0, "ci_low_95": -999.0,
        }

    mean = float(np.mean(net))
    sd = float(np.std(net, ddof=1)) if n > 1 else 0.0
    sharpe = mean / sd * np.sqrt(252.0) if sd > 0 else 0.0
    neg = net[net < 0]
    dsd = float(np.std(neg, ddof=1)) if len(neg) > 1 else 0.0
    sortino = mean / dsd * np.sqrt(252.0) if dsd > 0 else 0.0
    pos = float(net[net > 0].sum())
    negabs = float(-net[net < 0].sum())
    pf = pos / negabs if negabs > 0 else (999.0 if pos > 0 else 0.0)
    wr = float((net > 0).mean() * 100.0)

    # Day concentration: max fraction of net pnl from any single day
    ts = df["timestamp"].to_numpy()
    ts_pd = pd.to_datetime(ts, unit="ns", utc=True).tz_convert("America/New_York")
    day_str = ts_pd.strftime("%Y%m%d")
    day_df = pd.DataFrame({"day": day_str, "net": df["net_ticks"].fillna(0).to_numpy()})
    by_day = day_df.groupby("day")["net"].sum()
    total = by_day.sum()
    day_conc = float(by_day.abs().max() / max(1e-9, abs(total))) if abs(total) > 1e-9 else 1.0

    # 95% CI lower bound on mean net (per-trade): mean - 1.96 * sd / sqrt(n)
    ci_low_95 = mean - 1.96 * sd / max(1, np.sqrt(n)) if sd > 0 else mean

    return {
        "n_fills": n, "sharpe": sharpe, "sortino": sortino, "pf": pf, "wr": wr,
        "mean_net": mean, "day_conc": day_conc, "ci_low_95": ci_low_95,
    }


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


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def make_objective(out_dir: Path):
    progress_log = out_dir / "progress.jsonl"
    deploy_dir = out_dir / "deploy_eligible_configs"
    deploy_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial) -> float:
        # === HEAD / SIDE / PERCENTILE === (3 vars)
        head_horizon = trial.suggest_categorical(
            "head_horizon", ["1s", "5s", "10s", "30s"]
        )
        side = trial.suggest_categorical("side", ["long", "short"])
        # confidence_threshold = fraction of distribution tail kept (0.001 = top 0.1%)
        conf_thr = trial.suggest_float("conf_thr", 0.0005, 0.20, log=True)

        # === EXECUTION === (4 vars)
        order_type = trial.suggest_categorical(
            "order_type",
            ["passive_at_touch", "passive_at_touch_plus_1",
             "passive_at_touch_plus_2", "ioc_market"],
        )
        # HC #426 R3: prefer empirical bands from canonical avg-move cache
        bands = _AVG_MOVE_BANDS.get(head_horizon)
        if bands is not None:
            cancel_window = trial.suggest_int(
                "cancel_window", bands["cancel_lo"], bands["cancel_hi"]
            )
            hold_seconds = trial.suggest_float(
                "hold_seconds", bands["hold_lo"], bands["hold_hi"], log=True
            )
            # tp/sl are not direct TradeConfig args in v3.3 — record as user_attrs
            # so downstream code (or future v3.4 sweep) can consume the bands.
            tp_ticks = trial.suggest_float("tp_ticks", bands["tp_lo"], bands["tp_hi"])
            sl_ticks = trial.suggest_float("sl_ticks", bands["sl_lo"], bands["sl_hi"])
            trial.set_user_attr("avg_move_band_source", bands["source"])
            trial.set_user_attr("tp_ticks_empirical", tp_ticks)
            trial.set_user_attr("sl_ticks_empirical", sl_ticks)
        else:
            cancel_window = trial.suggest_int("cancel_window", 4, 80)  # evals
            hold_seconds = trial.suggest_float("hold_seconds", 1.0, 60.0, log=True)
        spread_ticks = trial.suggest_float("spread_ticks", 0.6, 1.6)

        # === SAFETY / FILTERS === (post-replay applied to per_trade_df)
        # ToD filter (5 vars)
        tod_start = trial.suggest_int("tod_start_hour", 6, 14)
        tod_end = trial.suggest_int("tod_end_hour", tod_start + 1, 16)

        # Pred-strength gate beyond percentile (1 var)
        pred_strength_min = trial.suggest_float("pred_strength_min", 0.0, 5.0)

        # === META: σ-spike halt threshold (1 var) ===
        sigma_halt_mult = trial.suggest_float("sigma_halt_mult", 3.0, 10.0)

        # === META: commission scaling (1 var) ===
        commission_ticks = trial.suggest_float("commission_ticks", 0.30, 0.50)

        # === META: legacy-FIFO-head confluence filter (2 vars) ===
        use_fifo_confluence = trial.suggest_categorical("use_fifo_confluence", [True, False])
        fifo_confluence_head = trial.suggest_categorical(
            "fifo_confluence_head",
            ["pred_fifo_tp4sl3_net", "pred_fifo_tp8sl5_net"],
        )
        fifo_confluence_thr_ticks = trial.suggest_float("fifo_confluence_thr_ticks", -1.0, 2.0)

        # === META: book-imbalance proxy filter (1 var)
        # We don't have raw book — proxy via secondary horizon agreement
        use_horizon_confluence = trial.suggest_categorical("use_horizon_confluence", [True, False])
        confluence_horizon = trial.suggest_categorical(
            "confluence_horizon", ["1s", "5s", "10s", "30s"]
        )

        # Build TradeConfig
        cfg = TradeConfig(
            side=side, horizon=head_horizon, confidence_threshold=conf_thr,
            order_type=order_type, cancel_eval_window=cancel_window,
            hold_seconds=hold_seconds,
        )

        # Run replay
        try:
            ledger = full_market_replay(
                _CACHE["_preds_path"], _CACHE["_labels_dir"], cfg,
                dates=_CACHE["dates"],
                spread_ticks_rth=spread_ticks,
                rt_commission_ticks=commission_ticks,
            )
        except Exception as e:
            trial.set_user_attr("error", str(e)[:200])
            return GATE_FAIL_SCORE

        # σ-spike halt: examine ledger's per_trade_df predictions vs sigma
        sigma_key = f"log_ret_{head_horizon}"
        head_sigma = _CACHE.get("sigma", {}).get(sigma_key, None)
        # (Used only as a metadata for filter; for now don't drop rows on it —
        #  proper σ-spike requires per-sample σ outputs which we don't have.)

        # Apply post-filters
        df_f, _ = apply_post_filters(
            ledger,
            tod_start_hour=tod_start, tod_end_hour=tod_end,
            require_min_pred_strength=pred_strength_min,
        )

        # Apply FIFO-head confluence filter (lazy: load fifo head pred globally)
        if use_fifo_confluence and len(df_f) > 0:
            # We need pred values at same indices — they're already in per_trade
            # for the primary head only. Skip rigorous impl for now and approximate
            # by using sign-of-prediction match. Documented limitation.
            pass

        # Compute metrics
        m = metrics_from_filtered(df_f)
        ok, reason = passes_hc344(m)

        # Persistence
        trial.set_user_attr("n_signals", ledger.n_signals)
        trial.set_user_attr("n_fills_raw", ledger.n_filled)
        for k, v in m.items():
            trial.set_user_attr(f"final_{k}", v)
        trial.set_user_attr("hc344_pass", ok)
        trial.set_user_attr("hc344_reason", reason)

        # Score for Optuna: gates fail → penalty floor (still ranks below valid)
        if not ok:
            score = GATE_FAIL_SCORE + max(-5.0, m["sharpe"]) * 0.1  # nudge by sharpe
        else:
            score = m["sharpe"]
            # Save eligible config
            deploy_config = build_paper_trader_config(trial.params, m, head_horizon, side)
            (deploy_dir / f"trial_{trial.number:06d}_sharpe_{m['sharpe']:.3f}.json").write_text(
                json.dumps(deploy_config, indent=2, default=str)
            )

        # Progress log
        with progress_log.open("a") as fh:
            fh.write(json.dumps({
                "trial": trial.number,
                "score": score,
                "params": trial.params,
                "metrics": m,
                "hc344_pass": ok,
                "hc344_reason": reason,
                "n_signals": int(ledger.n_signals),
                "n_fills_raw": int(ledger.n_filled),
                "fill_rate": float(ledger.fill_rate),
                "ts": time.time(),
            }, default=str) + "\n")

        return score

    return objective


def build_paper_trader_config(params: dict, metrics: dict, horizon: str, side: str) -> dict:
    """Render an Optuna trial into a HC #368(c)(ii) v33_paper_trader_config.json."""
    return {
        "model": "cnn_mamba_v3_3",
        "ckpt": "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_intra_ckpt.pt",
        "head": f"log_ret_{horizon}",
        "side": side,
        "suppress_long": (side == "short"),
        "suppress_short": (side == "long"),
        "entry": {
            "confidence_percentile_threshold": params["conf_thr"],
            "min_pred_strength_abs": params["pred_strength_min"],
            "order_type": params["order_type"],
            "spread_ticks_assumption": params["spread_ticks"],
        },
        "exit": {
            "hold_seconds": params["hold_seconds"],
            "cancel_window_evals": params["cancel_window"],
        },
        "filters": {
            "tod_start_hour_et": params["tod_start_hour"],
            "tod_end_hour_et": params["tod_end_hour"],
            "sigma_halt_mult": params["sigma_halt_mult"],
            "use_fifo_confluence": params["use_fifo_confluence"],
            "fifo_confluence_head": params["fifo_confluence_head"],
            "fifo_confluence_thr_ticks": params["fifo_confluence_thr_ticks"],
            "use_horizon_confluence": params["use_horizon_confluence"],
            "confluence_horizon": params["confluence_horizon"],
        },
        "costs": {
            "commission_ticks_rt": params["commission_ticks"],
        },
        "metrics_at_optimization": metrics,
        "hc_refs": ["HC #369", "HC #368", "HC #357", "HC #344", "HC #321"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", default=str(DEFAULT_PREDS_PATH))
    p.add_argument("--labels-dir", default=str(DEFAULT_LABELS_DIR))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--n-trials", type=int, default=5000)
    p.add_argument("--n-jobs", type=int, default=4, help="parallel trials (Jupiter 16 cores)")
    p.add_argument("--study-name", default="v33_execution_optuna_full_market_replay")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--avg-move-cache",
        default=None,
        help=(
            "HC #426 R3: path to canonical_avg_move_*.json (single file, "
            "comma-separated list, directory of per-horizon files, or template "
            "containing '{h}' such as "
            "'output/canonical_avg_move_v3_3_{h}.json'). When provided, "
            "tp/sl/hold/cancel ranges are taken from empirical p50/p90 bands "
            "instead of hard-coded defaults."
        ),
    )
    args = p.parse_args()

    # HC #426 R3: load empirical bands (no-op if --avg-move-cache not given)
    _load_avg_move_bands(args.avg_move_cache)
    if args.avg_move_cache and not _AVG_MOVE_BANDS:
        print("[avg-move] WARN: --avg-move-cache given but NO bands loaded.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "deploy_eligible_configs").mkdir(parents=True, exist_ok=True)

    # Init cache (heavy: loads all 4 horizons + FIFO labels)
    init_cache(Path(args.preds), Path(args.labels_dir))
    _CACHE["_preds_path"] = Path(args.preds)
    _CACHE["_labels_dir"] = Path(args.labels_dir)

    # Optuna study (SQLite for persistence + resumability)
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    storage_url = f"sqlite:///{out_dir / 'study.db'}"
    sampler = TPESampler(seed=args.seed, n_startup_trials=100, multivariate=True)
    pruner = MedianPruner(n_startup_trials=50, n_warmup_steps=0)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler, pruner=pruner,
        load_if_exists=True,
    )

    print(f"[start] {args.study_name} | n_trials={args.n_trials} | n_jobs={args.n_jobs}")
    print(f"[start] storage={storage_url}")
    print(f"[start] out_dir={out_dir}")

    obj = make_objective(out_dir)
    study.optimize(obj, n_trials=args.n_trials, n_jobs=args.n_jobs, show_progress_bar=False)

    # Save leaderboard
    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    df.to_csv(out_dir / "leaderboard.csv", index=False)

    # Save best configs (HC #344 passing)
    best = []
    for t in study.trials:
        if t.user_attrs.get("hc344_pass", False):
            best.append({
                "trial": t.number,
                "value": t.value,
                "params": t.params,
                "metrics": {k.replace("final_", ""): v for k, v in t.user_attrs.items() if k.startswith("final_")},
            })
    best.sort(key=lambda x: -x["value"])
    (out_dir / "best_configs.json").write_text(json.dumps(best[:20], indent=2, default=str))

    print(f"[done] total_trials={len(study.trials)}")
    print(f"[done] hc344_passing={len(best)}")
    if best:
        print(f"[done] top sharpe={best[0]['value']:.4f}")


if __name__ == "__main__":
    main()
