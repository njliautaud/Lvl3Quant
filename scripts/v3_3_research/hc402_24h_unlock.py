"""
HC #402 — 24-HOUR PRODUCTION-READY UNLOCK SCRIPT

Two analyses in one process (shared NPZ + label cache for speed):

  (A) Top-K canonical re-validation of Optuna leaderboard:
      - Read top 50 trials from study.db (sorted by reported Sharpe)
      - Re-evaluate EACH trial with rt_commission_ticks = 0.376 (FIXED, HC #392)
      - Enforce production gate day_conc <= 0.20 (HC #344 strict, not Optuna's 0.70 relaxed)
      - Output CSV with: trial, sampled_commission, original_sharpe, real_sharpe,
                          real_tk_per_fill, real_day_conc, real_n_fills,
                          hc344_strict_pass, hc344_relaxed_pass

  (B) Per-head MFE/MAE economics on v3.3 32 outputs:
      - For each prediction head + side (long/short)
      - Stratify all signals into 10 confidence deciles by |prediction|
      - For each decile, run full_market_replay (ioc_market, hold=5s) to get mfe_ticks/mae_ticks
      - Report: median_MFE, mean_MFE, pct_above_0.376 (passive-cost), pct_above_1.376 (market-cost)
      - Identifies which heads (if any) have economically tradeable signal even before queue effects

Output goes to /home/jupiter/Lvl3Quant/output/hc402_unlock_<timestamp>/

Usage:
  cd /home/jupiter/Lvl3Quant
  nohup python -u scripts/v3_3_research/hc402_24h_unlock.py > logs/hc402_unlock.log 2>&1 &
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(PROJ))

from scripts.v3_3_research.full_market_replay import (
    TradeConfig, TradeLedger, full_market_replay, ES_RT_COMMISSION_TICKS_DEFAULT,
)
from scripts.v3_3_research.v33_execution_optuna_full_market_replay import (
    apply_post_filters, metrics_from_filtered, passes_hc344,
    DEFAULT_PREDS_PATH, DEFAULT_LABELS_DIR,
    GATE_MIN_N_FILLS, GATE_MIN_PF, GATE_MIN_SHARPE, GATE_MIN_CI_LOW_95,
)

# ------ PATHS / CONFIG ------
OPTUNA_DIR = PROJ / "output" / "v33_execution_optuna_20260516_HC399followup"
STUDY_DB = OPTUNA_DIR / "study.db"
LEADERBOARD = OPTUNA_DIR / "leaderboard.csv"
OUT_DIR = PROJ / "output" / f"hc402_unlock_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_COMMISSION = 0.376  # HC #392 — DO NOT CHANGE
STRICT_DAY_CONC_GATE = 0.20   # HC #344 production gate
RELAXED_DAY_CONC_GATE = 0.70  # Optuna's sweep gate (for comparison)
TOP_K = 50                    # re-validate top 50 by reported Sharpe

# ------ HEADS (32) for MFE/MAE economics ------
# All prediction heads except mask_ columns and target_ columns.
ECON_HEADS = [
    "log_ret_1s", "log_ret_5s", "log_ret_10s", "log_ret_30s",
    "log_ret_60s", "log_ret_5min",
    "p_up_5s", "p_up_10s", "p_up_30s", "p_up_60s",
    "log_ret_10s_q10", "log_ret_10s_q50", "log_ret_10s_q90",
    "log_ret_30s_q10", "log_ret_30s_q50", "log_ret_30s_q90",
    "log_ret_60s_q10", "log_ret_60s_q50", "log_ret_60s_q90",
    "pred_mfe_30s_ticks", "pred_mae_30s_ticks",
    "pred_mfe_60s_ticks", "pred_mae_60s_ticks",
    "pred_time_to_mfe_secs",
    "p_reversal_15s", "p_reversal_30s", "p_reversal_60s",
    "pred_realized_vol_30s_ticks",
    "fifo_tp4sl3_net", "fifo_tp8sl5_net",
    "fifo_tp4sl3_hit_tp", "fifo_tp8sl5_hit_tp",
]
# Only the regression-style heads with a "natural" tick interpretation work
# with full_market_replay's percentile sweep (passive/market across horizons 1s/5s/10s/30s).
# So for MFE/MAE economics we use the FOUR primary horizon heads.
ECON_REPLAY_HEADS = ["1s", "5s", "10s", "30s"]


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ===========================================================================
# (A) Top-K canonical re-validation
# ===========================================================================
def fetch_topK_trials(db_path: Path, k: int = TOP_K) -> pd.DataFrame:
    """Pull top-K trials from Optuna study.db by their reported Sharpe (final_sharpe)."""
    print(f"[{_now()}] (A) Reading top-{k} trials from {db_path.name} ...")
    if LEADERBOARD.exists():
        df = pd.read_csv(LEADERBOARD)
    else:
        # Fall back to study.db (need sqlalchemy schema)
        con = sqlite3.connect(str(db_path))
        df = pd.read_sql(
            "SELECT t.trial_id, t.number, "
            "  MAX(CASE WHEN tp.param_name='head_horizon' THEN tp.param_value END) AS p_head, "
            "  MAX(CASE WHEN tp.param_name='side' THEN tp.param_value END) AS p_side "
            "FROM trials t JOIN trial_params tp ON t.trial_id=tp.trial_id GROUP BY t.trial_id",
            con,
        )
        con.close()
    # Filter to HC #344-passing (Optuna's relaxed pass), then sort by final_sharpe descending
    if "user_attrs_hc344_pass" in df.columns:
        df = df[df["user_attrs_hc344_pass"] == True]
    if "user_attrs_final_sharpe" in df.columns:
        df = df.sort_values("user_attrs_final_sharpe", ascending=False).head(k)
    print(f"[{_now()}] (A) {len(df)} top trials loaded.")
    return df.reset_index(drop=True)


def reval_one_trial(row: pd.Series, preds_path: Path, labels_dir: Path) -> dict:
    """Re-evaluate a single trial at canonical commission + production day_conc."""
    head_horizon = row["params_head_horizon"]
    side = row["params_side"]
    conf_thr = float(row["params_conf_thr"])
    order_type = row["params_order_type"]
    cancel_window = int(row["params_cancel_window"])
    hold_seconds = float(row["params_hold_seconds"])
    spread_ticks = float(row["params_spread_ticks"])
    tod_start = int(row["params_tod_start_hour"])
    tod_end = int(row["params_tod_end_hour"])
    pred_strength_min = float(row["params_pred_strength_min"])
    sampled_commission = float(row["params_commission_ticks"])

    cfg = TradeConfig(
        side=side, horizon=head_horizon, confidence_threshold=conf_thr,
        order_type=order_type, cancel_eval_window=cancel_window,
        hold_seconds=hold_seconds,
    )

    try:
        ledger = full_market_replay(
            preds_path, labels_dir, cfg,
            spread_ticks_rth=spread_ticks,
            rt_commission_ticks=CANONICAL_COMMISSION,  # <-- HC #392 FIXED
        )
    except Exception as e:
        return {
            "trial": int(row["number"]),
            "error": str(e)[:120],
            "real_sharpe": np.nan,
        }

    df_f, _ = apply_post_filters(
        ledger,
        tod_start_hour=tod_start, tod_end_hour=tod_end,
        require_min_pred_strength=pred_strength_min,
    )
    m = metrics_from_filtered(df_f)

    # Recompute gates at BOTH strict (0.20) and relaxed (0.70) day_conc
    def gate(day_conc_thr: float) -> bool:
        if m["n_fills"] < GATE_MIN_N_FILLS: return False
        if m["pf"] < GATE_MIN_PF: return False
        if m["day_conc"] > day_conc_thr: return False
        if m["sharpe"] < GATE_MIN_SHARPE: return False
        if m["ci_low_95"] < GATE_MIN_CI_LOW_95: return False
        return True

    strict_pass = gate(STRICT_DAY_CONC_GATE)
    relaxed_pass = gate(RELAXED_DAY_CONC_GATE)

    # Per-day fill count for diagnostic
    if not df_f.empty:
        ts_pd = pd.to_datetime(df_f["timestamp"].to_numpy(), unit="ns", utc=True).tz_convert("America/New_York")
        per_day = pd.Series(ts_pd.strftime("%Y%m%d")).value_counts().to_dict()
    else:
        per_day = {}

    return {
        "trial": int(row["number"]),
        "head_horizon": head_horizon,
        "side": side,
        "order_type": order_type,
        "sampled_commission": sampled_commission,
        "canonical_commission": CANONICAL_COMMISSION,
        "original_reported_sharpe": float(row["user_attrs_final_sharpe"]),
        "original_day_conc": float(row["user_attrs_final_day_conc"]),
        "real_sharpe": m["sharpe"],
        "real_sortino": m["sortino"],
        "real_pf": m["pf"],
        "real_wr": m["wr"],
        "real_tk_per_fill": m["mean_net"],
        "real_day_conc": m["day_conc"],
        "real_n_fills": m["n_fills"],
        "real_ci_low_95": m["ci_low_95"],
        "hc344_strict_pass": strict_pass,    # day_conc <= 0.20 PRODUCTION
        "hc344_relaxed_pass": relaxed_pass,  # day_conc <= 0.70 OPTUNA
        "n_fills_per_day_breakdown": json.dumps(per_day),
        "conf_thr": conf_thr,
        "hold_seconds": hold_seconds,
        "cancel_window": cancel_window,
        "tod_start": tod_start,
        "tod_end": tod_end,
        "pred_strength_min": pred_strength_min,
    }


def run_topK_reval() -> pd.DataFrame:
    print(f"[{_now()}] (A) === TOP-K CANONICAL RE-VALIDATION START ===")
    trials = fetch_topK_trials(STUDY_DB)
    if trials.empty:
        print(f"[{_now()}] (A) No trials to re-validate, aborting analysis A.")
        return pd.DataFrame()

    results = []
    t0 = time.time()
    for i, (_, row) in enumerate(trials.iterrows()):
        elapsed = time.time() - t0
        eta = (len(trials) - i) * (elapsed / max(1, i)) if i > 0 else 0
        print(f"[{_now()}] (A) Trial {i+1}/{len(trials)} (trial#{int(row['number'])}, "
              f"reported_sharpe={row['user_attrs_final_sharpe']:.2f}) — eta {eta/60:.1f}m")
        results.append(reval_one_trial(row, DEFAULT_PREDS_PATH, DEFAULT_LABELS_DIR))

    df = pd.DataFrame(results)
    out_csv = OUT_DIR / "topK_canonical_reval.csv"
    df.to_csv(out_csv, index=False)
    print(f"[{_now()}] (A) Saved {len(df)} re-validated rows to {out_csv}")

    # Summary
    n_strict = int(df["hc344_strict_pass"].sum())
    n_relaxed = int(df["hc344_relaxed_pass"].sum())
    print(f"[{_now()}] (A) STRICT (day_conc<=0.20) passers: {n_strict}/{len(df)}")
    print(f"[{_now()}] (A) RELAXED (day_conc<=0.70) passers: {n_relaxed}/{len(df)}")

    if n_strict > 0:
        winners = df[df["hc344_strict_pass"]].sort_values("real_sharpe", ascending=False)
        print(f"[{_now()}] (A) === STRICT WINNERS ===")
        print(winners[["trial", "head_horizon", "side", "order_type",
                       "real_sharpe", "real_tk_per_fill", "real_day_conc",
                       "real_n_fills"]].to_string(index=False))
        winners.to_csv(OUT_DIR / "topK_STRICT_WINNERS.csv", index=False)

    return df


# ===========================================================================
# (B) Per-head MFE/MAE economics
# ===========================================================================
def per_decile_mfe_mae(head: str, side: str,
                       preds_path: Path, labels_dir: Path) -> list[dict]:
    """For one (head, side), run full_market_replay with ALL signals (conf=1.0),
    then bucket by signal_percentile into 10 deciles. Report MFE/MAE economics."""
    cfg = TradeConfig(
        side=side, horizon=head, confidence_threshold=1.0,  # all signals
        order_type="ioc_market",                            # 100% fill, isolates economics
        cancel_eval_window=4, hold_seconds=5.0,
    )
    try:
        ledger = full_market_replay(preds_path, labels_dir, cfg)
    except Exception as e:
        print(f"[{_now()}]   (B) ERROR head={head} side={side}: {e}")
        return []

    df = ledger.per_trade_df
    df = df[df["filled"]].copy()
    if df.empty:
        return []

    # Sort by signal_percentile; decile 0 = least confident, decile 9 = most confident
    df = df.sort_values("signal_percentile", ascending=False).reset_index(drop=True)
    n = len(df)
    decile_size = n // 10
    rows = []
    for d in range(10):
        lo = d * decile_size
        hi = (d + 1) * decile_size if d < 9 else n
        bucket = df.iloc[lo:hi]
        if bucket.empty:
            continue
        mfe = bucket["mfe_ticks"].to_numpy(dtype=float)
        mae = bucket["mae_ticks"].to_numpy(dtype=float)
        net = bucket["net_ticks"].to_numpy(dtype=float)
        rows.append({
            "head": head, "side": side,
            "decile": d, "n": len(bucket),
            "median_MFE_ticks": float(np.nanmedian(mfe)),
            "mean_MFE_ticks": float(np.nanmean(mfe)),
            "median_MAE_ticks": float(np.nanmedian(mae)),
            "mean_MAE_ticks": float(np.nanmean(mae)),
            "pct_MFE_gt_0.376": float(np.mean(mfe > 0.376) * 100),  # passive-cost beat
            "pct_MFE_gt_1.376": float(np.mean(mfe > 1.376) * 100),  # market-cost beat
            "mean_NET_at_market": float(np.nanmean(net)),  # after 1.376 cost
            "median_NET_at_market": float(np.nanmedian(net)),
            "wr_net_gt_0": float(np.mean(net > 0) * 100),
        })
    return rows


def run_mfe_mae_economics() -> pd.DataFrame:
    print(f"[{_now()}] (B) === MFE/MAE ECONOMICS START ===")
    all_rows = []
    for head in ECON_REPLAY_HEADS:
        for side in ["long", "short"]:
            print(f"[{_now()}] (B) head={head} side={side}")
            all_rows.extend(per_decile_mfe_mae(head, side, DEFAULT_PREDS_PATH, DEFAULT_LABELS_DIR))

    df = pd.DataFrame(all_rows)
    out_csv = OUT_DIR / "v33_mfe_mae_economics.csv"
    df.to_csv(out_csv, index=False)
    print(f"[{_now()}] (B) Saved {len(df)} (head, side, decile) rows to {out_csv}")

    if not df.empty:
        # Top deciles report
        top = df[df["decile"] == 9].sort_values("median_MFE_ticks", ascending=False)
        print(f"[{_now()}] (B) === TOP-DECILE (most confident 10%) PER HEAD ===")
        print(top[["head", "side", "n", "median_MFE_ticks", "mean_MFE_ticks",
                   "pct_MFE_gt_0.376", "pct_MFE_gt_1.376",
                   "mean_NET_at_market", "wr_net_gt_0"]].to_string(index=False))
        top.to_csv(OUT_DIR / "v33_top_decile_summary.csv", index=False)

    return df


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print(f"[{_now()}] HC #402 24h UNLOCK SCRIPT STARTING")
    print(f"[{_now()}] OUT_DIR = {OUT_DIR}")
    print(f"[{_now()}] CANONICAL_COMMISSION = {CANONICAL_COMMISSION}")
    print(f"[{_now()}] STRICT_DAY_CONC_GATE = {STRICT_DAY_CONC_GATE}")
    print()

    t0 = time.time()
    df_reval = run_topK_reval()
    t1 = time.time()
    print(f"\n[{_now()}] (A) Done in {(t1-t0)/60:.1f} min\n")

    df_econ = run_mfe_mae_economics()
    t2 = time.time()
    print(f"\n[{_now()}] (B) Done in {(t2-t1)/60:.1f} min\n")

    # Final summary
    summary = {
        "output_dir": str(OUT_DIR),
        "n_trials_revalidated": len(df_reval),
        "n_strict_passers": int(df_reval["hc344_strict_pass"].sum()) if len(df_reval) else 0,
        "n_relaxed_passers": int(df_reval["hc344_relaxed_pass"].sum()) if len(df_reval) else 0,
        "n_econ_rows": len(df_econ),
        "duration_minutes": (t2 - t0) / 60,
        "canonical_commission": CANONICAL_COMMISSION,
        "strict_day_conc_gate": STRICT_DAY_CONC_GATE,
        "hc_refs": ["HC #344", "HC #392", "HC #397B", "HC #402"],
    }
    (OUT_DIR / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(f"[{_now()}] === ALL DONE ===")
    print(json.dumps(summary, indent=2))
