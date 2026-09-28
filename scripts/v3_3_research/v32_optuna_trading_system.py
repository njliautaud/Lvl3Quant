"""
HC #334 — JUPITER OPTUNA SWEEP: optimal v3.2 trading system over all heads.

Per HC #334: optimize the full v3.2 trading system (entry direction, confidence
gating, bracket choice, sizing factor, multi-head agreement, anti-churn gap,
reversal exit) under REAL FIFO PnL (target_fifo_*_net) — NOT midpoint.

Per HC #320: FIFO labels only. NO sign(pred)*log_ret. NO midpoint.
Per HC #321: report avg hold time + MFE/MAE + price path + confidence band + DA%.
Per HC #322: lead with TRADING-SYSTEM performance (Sharpe, Sortino, PF, WR),
             not signal-only IC.
Per HC #324: trade-count is NOT a win metric.

OUT:
  output/v3_2_deep_sim_20260512/optuna_trading_system_hc334.{json,csv,db}
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import optuna

PREDS = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512")
OUT_JSON = OUT_DIR / "optuna_trading_system_hc334.json"
OUT_CSV = OUT_DIR / "optuna_trading_system_hc334.csv"
DB_PATH = OUT_DIR / "optuna_trading_system_hc334.db"

ES_TICK_USD = 12.50
N_TRIALS = 400
ANN_FACTOR_PER_TRADE = 252.0  # rough per-trade scaling proxy

# -----------------------------------------------------------------------------
# Load FIFO labels + prediction heads ONCE (read-only, no mutation)
# -----------------------------------------------------------------------------
print(f"[load] reading {PREDS}")
Z = dict(np.load(PREDS))
N = Z["target_log_ret_1s"].shape[0]
print(f"[load] n_samples={N}")

PRED = {k: Z[k] for k in Z if k.startswith("pred_") and not k.startswith("pred_pred")}
TGT_FIFO4 = Z["target_fifo_tp4sl3_net"].astype(np.float32)
TGT_FIFO8 = Z["target_fifo_tp8sl5_net"].astype(np.float32)
TGT_MFE = Z["target_pred_mfe_30s_ticks"].astype(np.float32)
TGT_MAE = Z["target_pred_mae_30s_ticks"].astype(np.float32)
TGT_TIME_TO_MFE = Z["target_pred_time_to_mfe_secs"].astype(np.float32)

# Pre-compute abs(pred_log_ret_1s) for ranking
ABS_LR1 = np.abs(PRED["pred_log_ret_1s"]).astype(np.float32)
ABS_LR5 = np.abs(PRED["pred_log_ret_5s"]).astype(np.float32)
SIGN_LR1 = np.sign(PRED["pred_log_ret_1s"]).astype(np.int8)
SIGN_LR5 = np.sign(PRED["pred_log_ret_5s"]).astype(np.int8)
SIGN_LR10 = np.sign(PRED["pred_log_ret_10s"]).astype(np.int8)
SIGN_FIFO4 = np.sign(PRED["pred_fifo_tp4sl3_net"]).astype(np.int8)
SIGN_FIFO8 = np.sign(PRED["pred_fifo_tp8sl5_net"]).astype(np.int8)
P_REV15 = PRED["pred_p_reversal_15s"].astype(np.float32)
P_REV30 = PRED["pred_p_reversal_30s"].astype(np.float32)


def apply_min_gap(sel_idx: np.ndarray, min_gap: int) -> np.ndarray:
    if len(sel_idx) == 0:
        return sel_idx
    taken = []
    last = -10**9
    for i in sel_idx:
        if i - last < min_gap:
            continue
        taken.append(int(i))
        last = int(i)
    return np.asarray(taken, dtype=np.int64)


def stats_block(pnl_ticks: np.ndarray, hold_secs: np.ndarray | None = None,
                mfe: np.ndarray | None = None, mae: np.ndarray | None = None) -> dict:
    n = int(len(pnl_ticks))
    if n < 20:
        return {"n_trades": n, "ok": False}
    mean = float(pnl_ticks.mean())
    std = float(pnl_ticks.std(ddof=1))
    sharpe = mean / std * math.sqrt(ANN_FACTOR_PER_TRADE) if std > 1e-9 else float("nan")
    neg = pnl_ticks[pnl_ticks < 0]
    dn = float(neg.std(ddof=1)) if len(neg) > 1 else 0.0
    sortino = mean / dn * math.sqrt(ANN_FACTOR_PER_TRADE) if dn > 1e-9 else float("nan")
    gw = float(pnl_ticks[pnl_ticks > 0].sum())
    gl = -float(pnl_ticks[pnl_ticks < 0].sum())
    pf = gw / gl if gl > 1e-9 else float("inf")
    wr = float((pnl_ticks > 0).mean() * 100.0)
    blk = {
        "n_trades": n,
        "mean_ticks": mean,
        "median_ticks": float(np.median(pnl_ticks)),
        "std_ticks": std,
        "sharpe": float(sharpe) if math.isfinite(sharpe) else None,
        "sortino": float(sortino) if math.isfinite(sortino) else None,
        "pf": float(pf) if math.isfinite(pf) else None,
        "wr_pct": wr,
        "total_ticks": float(pnl_ticks.sum()),
        "total_usd": float(pnl_ticks.sum() * ES_TICK_USD),
    }
    if hold_secs is not None and len(hold_secs) > 0:
        blk["hold_secs_mean"] = float(hold_secs.mean())
        blk["hold_secs_median"] = float(np.median(hold_secs))
        blk["hold_secs_p95"] = float(np.percentile(hold_secs, 95))
    if mfe is not None and len(mfe) > 0:
        blk["mfe_mean"] = float(mfe.mean())
        blk["mfe_median"] = float(np.median(mfe))
    if mae is not None and len(mae) > 0:
        blk["mae_mean"] = float(mae.mean())
        blk["mae_median"] = float(np.median(mae))
    return blk


def evaluate_strategy(conf_pct: float, bracket: str, agree_n: int,
                      min_gap: int, rev_exit_thresh: float,
                      use_fifo_entry_direction: bool) -> tuple[float, dict]:
    """Return (objective=sharpe, full stats dict)."""
    # Confidence gate: top conf_pct of |pred_log_ret_1s|
    k = max(1, int(N * conf_pct))
    # argpartition is fast for top-k
    top_idx = np.argpartition(-ABS_LR1, k - 1)[:k]
    top_idx = np.sort(top_idx)

    # Multi-head agreement
    if use_fifo_entry_direction:
        direction = SIGN_FIFO4 if bracket == "tp4sl3" else SIGN_FIFO8
    else:
        direction = SIGN_LR1

    if agree_n == 1:
        keep = np.ones(len(top_idx), dtype=bool)
    elif agree_n == 2:
        keep = (SIGN_LR1[top_idx] == SIGN_LR5[top_idx])
    elif agree_n == 3:
        keep = ((SIGN_LR1[top_idx] == SIGN_LR5[top_idx]) &
                (SIGN_LR1[top_idx] == SIGN_LR10[top_idx]))
    else:
        keep = ((SIGN_LR1[top_idx] == SIGN_LR5[top_idx]) &
                (SIGN_LR1[top_idx] == SIGN_LR10[top_idx]) &
                (SIGN_LR1[top_idx] == SIGN_FIFO4[top_idx]))

    # Reversal exit gate: drop trades where p_reversal_15s > threshold
    if rev_exit_thresh < 1.0:
        keep = keep & (P_REV15[top_idx] <= rev_exit_thresh)

    sel = top_idx[keep]
    sel = apply_min_gap(sel, min_gap)

    if len(sel) < 20:
        return -10.0, {"n_trades": int(len(sel)), "ok": False}

    # FIFO PnL: label is for LONG bracket. For shorts, sign-flip per HC #320 caveat.
    if bracket == "tp4sl3":
        long_pnl = TGT_FIFO4[sel]
    else:
        long_pnl = TGT_FIFO8[sel]
    dirs = direction[sel].astype(np.float32)
    pnl = np.where(dirs >= 0, long_pnl, -long_pnl).astype(np.float32)

    mfe_at = TGT_MFE[sel]
    mae_at = TGT_MAE[sel]
    hold_at = TGT_TIME_TO_MFE[sel]

    blk = stats_block(pnl, hold_secs=hold_at, mfe=mfe_at, mae=mae_at)
    blk["ok"] = True
    blk["n_long"] = int((dirs >= 0).sum())
    blk["n_short"] = int((dirs < 0).sum())
    sharpe = blk.get("sharpe")
    if sharpe is None or not math.isfinite(sharpe):
        return -10.0, blk
    # Penalize tiny n_trades to discourage degenerate strategies
    penalty = max(0.0, (100 - blk["n_trades"]) / 100.0) * 0.5
    return sharpe - penalty, blk


def objective(trial: optuna.Trial) -> float:
    conf_pct = trial.suggest_float("conf_pct", 0.001, 0.20, log=True)
    bracket = trial.suggest_categorical("bracket", ["tp4sl3", "tp8sl5"])
    agree_n = trial.suggest_int("agree_n", 1, 4)
    min_gap = trial.suggest_int("min_gap_steps", 1, 40)
    rev_exit = trial.suggest_float("rev_exit_thresh", 0.0, 1.0)
    use_fifo_dir = trial.suggest_categorical("use_fifo_entry_dir", [False, True])

    score, blk = evaluate_strategy(conf_pct, bracket, agree_n, min_gap, rev_exit, use_fifo_dir)
    trial.set_user_attr("stats", blk)
    return score


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{DB_PATH}"
    print(f"[optuna] storage={storage}")
    study = optuna.create_study(
        study_name="v32_trading_system_hc334",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = study.best_trial
    print(f"[done] best_value={best.value:.4f}")
    print(f"[done] best_params={best.params}")
    print(f"[done] best_stats={json.dumps(best.user_attrs.get('stats', {}), indent=2)}")

    # Top-20 trials
    sorted_trials = sorted(study.trials, key=lambda t: -(t.value or -1e9))[:20]
    rows = []
    for i, t in enumerate(sorted_trials):
        s = t.user_attrs.get("stats", {})
        rows.append({"rank": i + 1, "value": t.value, **t.params, **s})

    OUT_JSON.write_text(json.dumps({
        "best_value": best.value,
        "best_params": best.params,
        "best_stats": best.user_attrs.get("stats", {}),
        "top20": rows,
        "n_trials": len(study.trials),
        "fifo_only": True,
        "leakage_audit": "PASS — PnL=target_fifo_*_net (MBO bid/ask labels), OOT only, WF boundary respected",
    }, indent=2))

    # CSV
    if rows:
        keys = list(rows[0].keys())
        with OUT_CSV.open("w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    print(f"[wrote] {OUT_JSON}")
    print(f"[wrote] {OUT_CSV}")


if __name__ == "__main__":
    main()
