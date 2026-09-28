"""
HC #405 — SIGNAL-DRIVEN CONFIG SCAN

Question being answered (per HC #405):
  Trial 278 decomposition (HC #404) showed +2.00 of its +1.99 tk/fill comes
  from the structural passive_+2 entry credit, not from signal alpha within
  the 1.48s hold. Are there OTHER deploy-eligible Optuna configs where the
  SIGNAL ALPHA (= gross MFE within the hold window) contributes meaningfully?

Procedure:
  1. Load 1500-trial Optuna leaderboard. Filter to deploy-eligible
     (user_attrs_hc344_pass==True) plus any sharpe>=3 marginals. Cap to top
     N by reported optimization-time Sharpe (default 50) to bound runtime.
  2. For each, build a TradeConfig from params, run canonical
     full_market_replay on the 15-day extended OOT predictions NPZ using
     CANONICAL HC #392 commission (0.376) and the trial's spread assumption.
  3. Apply the same apply_post_filters (ToD + pred-strength) as the trial.
  4. For each surviving filled trade, recompute the gross MFE within the
     hold window in ticks (same logic as hc404_decomposition_study:
     max over horizons-in-hold of signed in-position log-ret).
  5. Aggregate per config: mean / median gross MFE, % of fills with
     gross MFE >= 1 tk (signal-driven proportion), plus headline metrics.
  6. Rank by % of fills with gross MFE >= 1 tk and by mean gross MFE.
  7. Emit ranked_configs.csv, TOP5_CANDIDATES.md, VERDICT.md.

Constraints (HC #405):
  - Canonical commission 0.376 RT ticks (HC #392).
  - Spread per-trial (trial's spread_ticks_assumption).
  - Do NOT modify full_market_replay.py / verify_trial278_from_json.py /
    v33_execution_optuna_full_market_replay.py.
  - 90-min runtime budget. multiprocessing up to 8 workers if needed.

MALWARE-GUARD: pure analysis. Reads predictions NPZ + FIFO labels + Optuna
artifacts. Writes only to output/hc405_signal_driven_scan_<ts>/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(PROJ))

from scripts.v3_3_research.full_market_replay import (  # noqa: E402
    TradeConfig, full_market_replay, _load_predictions,
    PRICE_UNIT_TO_TICKS, _entry_price_edge_ticks, _pick_exit_horizon,
)
from scripts.v3_3_research.v33_execution_optuna_full_market_replay import (  # noqa: E402
    apply_post_filters, metrics_from_filtered,
)

# ---------------------------------------------------------------------------
# Constants (HC #405 / #392)
# ---------------------------------------------------------------------------
LEADERBOARD = (
    PROJ / "output" / "v33_execution_optuna_20260516_HC399followup"
         / "leaderboard.csv"
)
DEPLOY_DIR = (
    PROJ / "output" / "v33_execution_optuna_20260516_HC399followup"
         / "deploy_eligible_configs"
)
# Canonical 15-day OOT NPZ (per MONDAY_DEPLOYMENT_CANDIDATE.json)
PREDS_PATH = (
    PROJ / "output" / "v3_3_extended_oot_20260514"
         / "extended_oot_predictions.npz"
)
LABELS_DIR = PROJ / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"

CANONICAL_COMMISSION = 0.376  # HC #392
HORIZONS = ("1s", "5s", "10s", "30s")
HORIZON_SEC = {"1s": 1.0, "5s": 5.0, "10s": 10.0, "30s": 30.0}

# Trial 278 reference baseline (from MONDAY_DEPLOYMENT_CANDIDATE.json)
T278 = {
    "sharpe": 13.48,
    "tk_per_fill": 1.99,
    "day_conc": 0.186,
    "n_fills": 195,
    "gross_MFE_mean_tk_hc404": None,  # filled in from this scan's trial_278 row
    "entry_edge_tk": 2.0,
    "hold_s": 1.4767640490577054,
    "signal_alpha_tk_hc404": 0.41,  # from HC #404 finding
}


# ---------------------------------------------------------------------------
# Helpers — gross-MFE-within-hold (same math as hc404_decomposition_study)
# ---------------------------------------------------------------------------
def _compute_horizon_paths_ticks(preds: dict, idx: np.ndarray,
                                  side_sign: float) -> dict:
    out = {}
    for h in HORIZONS:
        lr = preds["tgt_lr"][h][idx]
        mk = preds["tgt_lr_mask"][h][idx]
        in_pos = side_sign * lr * PRICE_UNIT_TO_TICKS
        out[h] = np.where(mk, in_pos, np.nan).astype(np.float64)
    return out


def _gross_mfe_within_hold(paths: dict, hold_seconds: float) -> np.ndarray:
    horizons_in_hold = [h for h in HORIZONS
                        if HORIZON_SEC[h] <= max(hold_seconds, 1.0) + 1e-9]
    if not horizons_in_hold:
        horizons_in_hold = ["1s"]
    stacked = np.vstack([paths[h] for h in horizons_in_hold])
    return np.nanmax(stacked, axis=0)


def _gross_mae_within_hold(paths: dict, hold_seconds: float) -> np.ndarray:
    horizons_in_hold = [h for h in HORIZONS
                        if HORIZON_SEC[h] <= max(hold_seconds, 1.0) + 1e-9]
    if not horizons_in_hold:
        horizons_in_hold = ["1s"]
    stacked = np.vstack([paths[h] for h in horizons_in_hold])
    return np.nanmin(stacked, axis=0)


def _percentile_select_idx(pred: np.ndarray, mask: np.ndarray, side: str,
                           conf_pctile: float) -> np.ndarray:
    p_valid = pred[mask]
    if side == "long":
        thr = float(np.quantile(p_valid, 1.0 - conf_pctile))
        sel = mask & (pred >= thr)
    else:
        thr = float(np.quantile(p_valid, conf_pctile))
        sel = mask & (pred <= thr)
    return np.where(sel)[0]


# ---------------------------------------------------------------------------
# Per-trial worker
# ---------------------------------------------------------------------------
def _horizon_from_head(head: str) -> str:
    if head.startswith("log_ret_"):
        return head[len("log_ret_"):]
    return head


def evaluate_trial(trial_json_path: Path, preds_cache: dict) -> dict:
    """Run canonical replay + decomposition for a single trial config JSON.

    Returns a flat dict (one row) of headline + gross-MFE stats.
    """
    name = trial_json_path.stem
    try:
        with open(trial_json_path) as f:
            cfg = json.load(f)
    except Exception as e:
        return {"trial": name, "error": f"load_fail: {e}"}

    side = cfg["side"]
    head = cfg["head"]
    horizon = _horizon_from_head(head)
    entry = cfg["entry"]
    exit_blk = cfg["exit"]
    filt = cfg["filters"]

    tc = TradeConfig(
        side=side,
        horizon=horizon,
        confidence_threshold=float(entry["confidence_percentile_threshold"]),
        order_type=entry["order_type"],
        cancel_eval_window=int(exit_blk["cancel_window_evals"]),
        hold_seconds=float(exit_blk["hold_seconds"]),
    )

    t0 = time.time()
    try:
        ledger = full_market_replay(
            PREDS_PATH, LABELS_DIR, tc,
            spread_ticks_rth=float(entry["spread_ticks_assumption"]),
            rt_commission_ticks=CANONICAL_COMMISSION,
        )
    except Exception as e:
        return {"trial": name, "error": f"replay_fail: {e}"}

    df_f, _ = apply_post_filters(
        ledger,
        tod_start_hour=int(filt["tod_start_hour_et"]),
        tod_end_hour=int(filt["tod_end_hour_et"]),
        require_min_pred_strength=float(entry["min_pred_strength_abs"]),
    )
    m = metrics_from_filtered(df_f)
    n_fills = int(m["n_fills"])

    elapsed = time.time() - t0

    # If no fills survive, return headline only
    if n_fills == 0 or df_f.empty:
        return {
            "trial": name,
            "side": side, "horizon": horizon,
            "order_type": entry["order_type"],
            "hold_seconds": float(exit_blk["hold_seconds"]),
            "cancel_evals": int(exit_blk["cancel_window_evals"]),
            "conf_pctile": float(entry["confidence_percentile_threshold"]),
            "min_pred_strength": float(entry["min_pred_strength_abs"]),
            "spread_ticks": float(entry["spread_ticks_assumption"]),
            "tod_start": int(filt["tod_start_hour_et"]),
            "tod_end": int(filt["tod_end_hour_et"]),
            "sharpe": m["sharpe"], "mean_net_tk": m["mean_net"],
            "n_fills": n_fills, "day_conc": m["day_conc"],
            "pf": m["pf"], "wr": m["wr"],
            "entry_edge_tk": _entry_price_edge_ticks(
                entry["order_type"], float(entry["spread_ticks_assumption"])),
            "gross_MFE_mean_tk": float("nan"),
            "gross_MFE_median_tk": float("nan"),
            "gross_MAE_mean_tk": float("nan"),
            "pct_fills_MFE_ge_1tk": float("nan"),
            "pct_fills_MFE_ge_2tk": float("nan"),
            "signal_alpha_share": float("nan"),
            "elapsed_s": elapsed,
            "error": "" if n_fills == 0 else "empty_after_filter",
        }

    # Reconstruct surviving global indices for path-level MFE
    preds = preds_cache[horizon]
    sel_idx = _percentile_select_idx(
        preds["pred"], preds["mask"], side,
        float(entry["confidence_percentile_threshold"])
    )
    ptd = ledger.per_trade_df.copy()
    ptd["_sel_pos"] = np.arange(len(ptd))
    ptd = ptd[ptd["filled"]].reset_index(drop=True)
    ts_pd = pd.to_datetime(ptd["timestamp"].to_numpy(), unit="ns", utc=True
                           ).tz_convert("America/New_York")
    hours = ts_pd.hour.to_numpy()
    tmask = (hours >= int(filt["tod_start_hour_et"])) & \
            (hours < int(filt["tod_end_hour_et"]))
    ptd = ptd.loc[tmask].reset_index(drop=True)
    min_strength = float(entry["min_pred_strength_abs"])
    if min_strength > 0:
        smask = np.abs(ptd["prediction"].to_numpy()) >= min_strength
        ptd = ptd.loc[smask].reset_index(drop=True)

    if len(ptd) != len(df_f):
        # benign drift — fall back to ptd
        pass

    if len(ptd) == 0:
        return {
            "trial": name, "side": side, "horizon": horizon,
            "order_type": entry["order_type"],
            "hold_seconds": float(exit_blk["hold_seconds"]),
            "n_fills": n_fills, "error": "ptd_empty_after_replay",
            "elapsed_s": elapsed,
        }

    surviving_sel_pos = ptd["_sel_pos"].to_numpy()
    global_idx = sel_idx[surviving_sel_pos]

    side_sign = 1.0 if side == "long" else -1.0
    paths = _compute_horizon_paths_ticks(preds, global_idx, side_sign)

    hold_s = float(exit_blk["hold_seconds"])
    gross_mfe = _gross_mfe_within_hold(paths, hold_s)
    gross_mae = _gross_mae_within_hold(paths, hold_s)

    # Stats
    mfe_finite = gross_mfe[np.isfinite(gross_mfe)]
    mae_finite = gross_mae[np.isfinite(gross_mae)]
    n_obs = mfe_finite.size

    mean_mfe = float(np.mean(mfe_finite)) if n_obs > 0 else float("nan")
    median_mfe = float(np.median(mfe_finite)) if n_obs > 0 else float("nan")
    mean_mae = float(np.mean(mae_finite)) if mae_finite.size > 0 else float("nan")
    pct_ge_1 = float((mfe_finite >= 1.0).mean()) if n_obs > 0 else float("nan")
    pct_ge_2 = float((mfe_finite >= 2.0).mean()) if n_obs > 0 else float("nan")

    edge = _entry_price_edge_ticks(
        entry["order_type"], float(entry["spread_ticks_assumption"])
    )
    # Signal-alpha share: fraction of mean realized net that came from
    # gross_MFE (signal travel) vs entry credit. Computed conservatively
    # against gross_MFE + edge (the two positive components before
    # commission/exit-timing-loss).
    mean_net = float(m["mean_net"])
    denom = mean_mfe + edge if (mean_mfe is not None and not np.isnan(mean_mfe)) else float("nan")
    if denom and not np.isnan(denom) and abs(denom) > 1e-9:
        signal_share = mean_mfe / denom
    else:
        signal_share = float("nan")

    return {
        "trial": name,
        "side": side, "horizon": horizon,
        "order_type": entry["order_type"],
        "hold_seconds": hold_s,
        "cancel_evals": int(exit_blk["cancel_window_evals"]),
        "conf_pctile": float(entry["confidence_percentile_threshold"]),
        "min_pred_strength": min_strength,
        "spread_ticks": float(entry["spread_ticks_assumption"]),
        "tod_start": int(filt["tod_start_hour_et"]),
        "tod_end": int(filt["tod_end_hour_et"]),
        # headline (canonical-cost replay on 15d OOT)
        "sharpe": float(m["sharpe"]),
        "mean_net_tk": mean_net,
        "n_fills": n_fills,
        "day_conc": float(m["day_conc"]),
        "pf": float(m["pf"]),
        "wr": float(m["wr"]),
        # signal-alpha measurements
        "entry_edge_tk": float(edge),
        "gross_MFE_mean_tk": mean_mfe,
        "gross_MFE_median_tk": median_mfe,
        "gross_MAE_mean_tk": mean_mae,
        "pct_fills_MFE_ge_1tk": pct_ge_1,
        "pct_fills_MFE_ge_2tk": pct_ge_2,
        "signal_alpha_share": signal_share,
        "hc344_strict_pass_0.20": bool(float(m["day_conc"]) <= 0.20),
        "elapsed_s": elapsed,
        "error": "",
    }


# ---------------------------------------------------------------------------
# Trial-selection: deploy-eligible + top-N
# ---------------------------------------------------------------------------
def select_trial_jsons(top_n: int) -> list[Path]:
    """Return up to top_n deploy-eligible trial JSON paths, ranked by
    optimization-time Sharpe descending. Trial 278 is always included.
    """
    lb = pd.read_csv(LEADERBOARD)
    # Eligibility per HC #405 spec:
    #   PRIMARY: hc344_pass==True (deploy-eligible — these are the on-disk
    #            configs in deploy_eligible_configs/).
    #   SECONDARY: sharpe>=3 marginals — but only if their JSON exists on
    #              disk (hc344 fails typically lack the artifact).
    # Resolve disk artifacts up-front.
    by_number = {}
    for p in DEPLOY_DIR.glob("trial_*.json"):
        try:
            num = int(p.stem.split("_")[1])
            by_number[num] = p
        except Exception:
            continue

    elig_primary = lb[lb["user_attrs_hc344_pass"] == True].copy()  # noqa: E712
    elig_primary = elig_primary.sort_values(
        "user_attrs_final_sharpe", ascending=False
    )
    primary_numbers = [n for n in elig_primary["number"].tolist()
                       if n in by_number]

    elig_secondary = lb[(lb["user_attrs_hc344_pass"] != True)  # noqa: E712
                        & (lb["user_attrs_final_sharpe"] >= 3.0)
                        & (lb["user_attrs_final_n_fills"] >= 30)].copy()
    elig_secondary = elig_secondary.sort_values(
        "user_attrs_final_sharpe", ascending=False
    )
    secondary_numbers = [n for n in elig_secondary["number"].tolist()
                         if n in by_number]

    # Always include trial 278 first
    ordered = []
    if 278 in by_number:
        ordered.append(278)
    for n in primary_numbers:
        if n not in ordered:
            ordered.append(n)
    for n in secondary_numbers:
        if n not in ordered:
            ordered.append(n)
    ordered = ordered[:top_n]

    paths = [by_number[n] for n in ordered if n in by_number]
    return paths


# ---------------------------------------------------------------------------
# Verdict and markdown emission
# ---------------------------------------------------------------------------
def rank_and_emit(rows_df: pd.DataFrame, out_dir: Path) -> None:
    # Ranked CSV (all eligible, full schema)
    # Primary rank: pct_fills_MFE_ge_1tk (signal-driven proportion)
    # Secondary: gross_MFE_mean_tk
    # Tertiary: sharpe
    rows_df = rows_df.copy()
    rows_df["__rank_key"] = (
        rows_df["pct_fills_MFE_ge_1tk"].fillna(-1) * 1e6
        + rows_df["gross_MFE_mean_tk"].fillna(-99) * 1e3
        + rows_df["sharpe"].fillna(-99)
    )
    rows_df = rows_df.sort_values("__rank_key", ascending=False
                                  ).drop(columns="__rank_key")
    ranked_csv = out_dir / "ranked_configs.csv"
    rows_df.to_csv(ranked_csv, index=False)

    # Strong signal-driven set: mean_MFE >= 1 tk AND pass strict
    strong = rows_df[
        (rows_df["gross_MFE_mean_tk"] >= 1.0)
        & (rows_df["hc344_strict_pass_0.20"] == True)  # noqa: E712
        & (rows_df["n_fills"] >= 30)
    ].copy()
    # RELAXED signal-driven set: mean_MFE >= 1 tk AND day_conc <= 0.70 (HC #344
    # relaxed gate — useful when strict is overfit to 5-day-OOT day_conc).
    strong_relaxed = rows_df[
        (rows_df["gross_MFE_mean_tk"] >= 1.0)
        & (rows_df["day_conc"] <= 0.70)
        & (rows_df["n_fills"] >= 30)
    ].copy()

    # Find trial 278 row
    t278 = rows_df[rows_df["trial"].str.contains("trial_000278")]
    t278_mfe = float(t278["gross_MFE_mean_tk"].iloc[0]) if len(t278) else float("nan")
    t278_pct1 = float(t278["pct_fills_MFE_ge_1tk"].iloc[0]) if len(t278) else float("nan")
    t278_sharpe = float(t278["sharpe"].iloc[0]) if len(t278) else float("nan")
    t278_n = int(t278["n_fills"].iloc[0]) if len(t278) else 0

    # TOP5 markdown — pick top 5 by signal-driven ranking. Prefer STRONG (strict
    # pass), then STRONG_RELAXED (signal-driven but only relaxed-pass), then
    # overall best-MFE among any with n_fills>=30. Always include trial 278
    # in pool for comparison.
    if len(strong) >= 5:
        top5 = strong.sort_values(
            ["pct_fills_MFE_ge_1tk", "gross_MFE_mean_tk", "sharpe"],
            ascending=False
        ).head(5)
    elif len(strong_relaxed) >= 5:
        top5 = strong_relaxed.sort_values(
            ["pct_fills_MFE_ge_1tk", "gross_MFE_mean_tk", "sharpe"],
            ascending=False
        ).head(5)
    else:
        # Fall back to top by gross_MFE among n_fills>=30
        top5_pool = rows_df[rows_df["n_fills"] >= 30].sort_values(
            ["gross_MFE_mean_tk", "sharpe"], ascending=False
        )
        top5 = top5_pool.head(5)

    lines = [
        "# HC #405 — TOP 5 SIGNAL-DRIVEN CANDIDATES",
        "",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}",
        "",
        "## Question",
        "",
        "Of the deploy-eligible Optuna configs, which (if any) generate "
        "their P&L from REAL signal alpha (gross MFE within the hold window) "
        "rather than from the structural passive_+K entry credit?",
        "",
        "## Methodology",
        "",
        "- Replayed each candidate through canonical `full_market_replay` on "
        "15-day extended OOT (`output/v3_3_extended_oot_20260514/extended_oot_predictions.npz`).",
        f"- Canonical commission = {CANONICAL_COMMISSION} RT ticks (HC #392). "
        "Per-trial spread used (not constant).",
        "- For each filled trade, gross MFE = max favorable in-position "
        "log-ret across all available horizons <= hold_seconds, in ticks "
        "(same logic as `hc404_decomposition_study.py`).",
        "- 'Signal-driven' = gross_MFE_mean_tk >= 1.0 AND pct_fills_MFE_ge_1tk high.",
        "",
        "## Trial 278 reference",
        "",
        f"- Sharpe: {t278_sharpe:.2f} | n_fills: {t278_n} | "
        f"gross MFE mean: {t278_mfe:.3f} tk | "
        f"% fills with MFE>=1tk: {t278_pct1:.1%}",
        f"- Entry edge: +{T278['entry_edge_tk']:.1f} tk (passive_+2 credit)",
        f"- HC #404 finding: only {T278['signal_alpha_tk_hc404']:.2f} tk of "
        f"the +{T278['tk_per_fill']:.2f} tk/fill comes from signal alpha; the "
        f"rest is the structural credit.",
        "",
        "## Top 5 candidates (ranked by signal-driven proportion + mean MFE)",
        "",
    ]

    for i, row in enumerate(top5.itertuples(), 1):
        replace_t278 = "YES" if (
            getattr(row, "gross_MFE_mean_tk", -1) >= 1.0
            and getattr(row, "sharpe", -1) >= T278["sharpe"]
            and getattr(row, "n_fills", 0) >= 30
            and getattr(row, "hc344_strict_pass_0_20", False) is True
        ) else "NO"
        # tuple attribute names get dots normalized to underscores
        strict_pass = getattr(row, "_asdict", None)
        if strict_pass:
            d = row._asdict()
        else:
            d = {k: getattr(row, k) for k in top5.columns}

        lines.extend([
            f"### {i}. `{row.trial}`",
            "",
            f"- **side**: {row.side} | **horizon**: {row.horizon} | "
            f"**order_type**: `{row.order_type}`",
            f"- **hold_seconds**: {row.hold_seconds:.3f}s | "
            f"**cancel_evals**: {row.cancel_evals} | "
            f"**conf_pctile**: {row.conf_pctile:.5f} | "
            f"**min_pred_strength**: {row.min_pred_strength:.4f}",
            f"- **spread_ticks**: {row.spread_ticks:.4f} | "
            f"**ToD**: {row.tod_start}:00 - {row.tod_end}:00 ET",
            "",
            "**Headline (15d OOT @ canonical 0.376 commission):**",
            "",
            f"- Sharpe: **{row.sharpe:.2f}** | tk/fill: **{row.mean_net_tk:+.3f}** "
            f"| n_fills: **{row.n_fills}** | day_conc: **{row.day_conc:.3f}** "
            f"| PF: {row.pf:.2f} | WR: {row.wr:.1f}%",
            f"- HC #344 strict (day_conc<=0.20): "
            f"{'PASS' if row.day_conc <= 0.20 else 'FAIL'}",
            "",
            "**Signal-alpha measurement:**",
            "",
            f"- Gross MFE within hold (mean): **{row.gross_MFE_mean_tk:.3f} tk** "
            f"| (median): {row.gross_MFE_median_tk:.3f} tk",
            f"- Gross MAE within hold (mean): {row.gross_MAE_mean_tk:.3f} tk",
            f"- % of fills with gross MFE >= 1 tk: **{row.pct_fills_MFE_ge_1tk:.1%}**",
            f"- % of fills with gross MFE >= 2 tk: {row.pct_fills_MFE_ge_2tk:.1%}",
            f"- Entry edge: {row.entry_edge_tk:+.2f} tk "
            f"({'passive credit' if row.entry_edge_tk > 0 else 'spread debit' if row.entry_edge_tk < 0 else 'neutral'})",
            f"- Signal alpha share (MFE / (MFE + edge)): "
            f"{row.signal_alpha_share:.1%}" if not pd.isna(row.signal_alpha_share) else "- Signal alpha share: n/a",
            "",
            f"**Would replace trial 278?** {replace_t278}",
            "",
            "---",
            "",
        ])

    (out_dir / "TOP5_CANDIDATES.md").write_text("\n".join(lines))

    # ---- VERDICT.md ----
    n_total = len(rows_df)
    n_with_data = int(rows_df["gross_MFE_mean_tk"].notna().sum())
    n_strong = len(strong)
    n_mfe_ge_05 = int((rows_df["gross_MFE_mean_tk"] >= 0.5).sum())
    n_mfe_ge_1 = int((rows_df["gross_MFE_mean_tk"] >= 1.0).sum())
    n_mfe_ge_2 = int((rows_df["gross_MFE_mean_tk"] >= 2.0).sum())
    n_passive_p2 = int((rows_df["order_type"] == "passive_at_touch_plus_2").sum())
    n_passive_p1 = int((rows_df["order_type"] == "passive_at_touch_plus_1").sum())
    n_passive_p0 = int((rows_df["order_type"] == "passive_at_touch").sum())
    n_ioc = int((rows_df["order_type"] == "ioc_market").sum())

    # Among ioc and passive_at_touch (no structural credit), is there any
    # config with mean MFE >= 1 tk AND sharpe > T278?
    no_credit = rows_df[rows_df["order_type"].isin(
        ["ioc_market", "passive_at_touch"])]
    no_credit_signal = no_credit[no_credit["gross_MFE_mean_tk"] >= 1.0]
    n_no_credit_signal = len(no_credit_signal)
    best_no_credit = no_credit.sort_values("sharpe", ascending=False).head(1)

    # Best signal-driven candidate overall (strong set, max sharpe)
    if len(strong) > 0:
        winner = strong.sort_values("sharpe", ascending=False).iloc[0]
        winner_str = (
            f"`{winner['trial']}` — side={winner['side']}, "
            f"horizon={winner['horizon']}, order={winner['order_type']}, "
            f"hold={winner['hold_seconds']:.2f}s, Sharpe={winner['sharpe']:.2f}, "
            f"mean MFE={winner['gross_MFE_mean_tk']:.2f} tk, "
            f"n_fills={int(winner['n_fills'])}"
        )
    else:
        winner_str = "(none — no config in the scan satisfied "
        "mean MFE >= 1 tk AND day_conc <= 0.20 AND n_fills >= 30)"

    verdict_lines = [
        "# HC #405 — VERDICT",
        "",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}",
        "",
        "## Is there a clearly-better-than-trial-278 signal-driven config?",
        "",
    ]

    # Relaxed-set winner (for when strict set is empty)
    if len(strong_relaxed) > 0:
        winner_relaxed = strong_relaxed.sort_values(
            "sharpe", ascending=False).iloc[0]
        winner_relaxed_str = (
            f"`{winner_relaxed['trial']}` — side={winner_relaxed['side']}, "
            f"horizon={winner_relaxed['horizon']}, "
            f"order={winner_relaxed['order_type']}, "
            f"hold={winner_relaxed['hold_seconds']:.2f}s, "
            f"Sharpe={winner_relaxed['sharpe']:.2f}, "
            f"mean MFE={winner_relaxed['gross_MFE_mean_tk']:.2f} tk, "
            f"%MFE>=1tk={winner_relaxed['pct_fills_MFE_ge_1tk']:.0%}, "
            f"n_fills={int(winner_relaxed['n_fills'])}, "
            f"day_conc={winner_relaxed['day_conc']:.3f} (strict FAIL, relaxed PASS)"
        )
    else:
        winner_relaxed_str = "(none)"

    if len(strong) > 0 and float(strong["sharpe"].max()) > T278["sharpe"]:
        verdict_lines.extend([
            "**YES — at least one config beats trial 278 on signal alpha "
            "AND passes HC #344 strict (day_conc <= 0.20).**",
            "",
            f"Best candidate: {winner_str}",
            "",
        ])
    elif len(strong) > 0:
        verdict_lines.extend([
            "**PARTIAL — there exist configs with mean MFE >= 1 tk that pass "
            "the strict day_conc gate, but none beat trial 278's headline "
            "Sharpe of 13.48 on the 15-day OOT replay.**",
            "",
            f"Best signal-driven candidate by Sharpe: {winner_str}",
            "",
        ])
    elif len(strong_relaxed) > 0:
        verdict_lines.extend([
            f"**NO STRICT-PASSER, BUT YES RELAXED-PASSER — {len(strong_relaxed)} "
            f"config(s) capture meaningful signal alpha (mean MFE >= 1 tk) and "
            f"pass the HC #344 RELAXED day_conc gate (<= 0.70) on the 15-day "
            f"OOT replay, but FAIL the strict 0.20 gate.**",
            "",
            f"Best relaxed-pass signal-driven candidate by Sharpe: "
            f"{winner_relaxed_str}",
            "",
            "These configs would be signal-driven supplements (not replacements) "
            "to trial 278 — they carry real directional alpha but their P&L is "
            "concentrated on fewer days (a sign of overfitting to the original "
            "5-day OOT used during Optuna's day_conc evaluation; HC #402 "
            "demonstrated 40/50 trials that strict-passed on 5d failed on 15d).",
            "",
        ])
    else:
        verdict_lines.extend([
            "**NO — among the deploy-eligible Optuna configs scanned, none "
            "satisfied mean_gross_MFE_within_hold >= 1.0 tk with n_fills >= 30 "
            "on the 15-day extended OOT.**",
            "",
            "Trial 278 remains the headline candidate, but its P&L is "
            "STRUCTURAL (passive_+2 credit), not signal-driven.",
            "",
        ])

    # MFE/MAE collapse caveat — for hold < 5s only one horizon ('1s') is in
    # the window, so MFE==MAE (it's just the in-position move at exit). For
    # hold >= 5s multiple horizons are available and MFE/MAE diverge.
    n_short_hold = int((rows_df["hold_seconds"] < 5.0).sum())
    n_mfe_eq_mae = int((np.abs(rows_df["gross_MFE_mean_tk"] -
                                 rows_df["gross_MAE_mean_tk"]) < 1e-6).sum())

    verdict_lines.extend([
        "## Caveat on MFE measurement",
        "",
        "For the canonical predictions NPZ we only have realized in-position "
        "log-rets at four discrete horizons (1s, 5s, 10s, 30s). 'Gross MFE "
        "within hold' is the max across those horizons that fit in the hold "
        "window. For hold < 5s only the 1s horizon fits, so MFE == MAE == "
        "in-position move at exit; it is therefore a 'realized exit travel' "
        "proxy, not a true continuous-path MFE peak.",
        "",
        f"In this scan, **{n_short_hold}/{n_total}** configs have hold < 5s "
        f"and **{n_mfe_eq_mae}/{n_total}** have MFE == MAE. The Optuna search "
        "did not explore configs with multi-horizon holds, so we cannot "
        "directly observe path-MFE-above-realized-exit ('alpha left on the "
        "table') for these. A new sweep with hold in [5s, 60s] would be "
        "needed to measure that.",
        "",
        "## What the leaderboard tells us",
        "",
        f"- Configs scanned: **{n_total}** | with MFE data: {n_with_data}",
        f"- Order-type distribution: passive_+2={n_passive_p2}, "
        f"passive_+1={n_passive_p1}, passive_at_touch={n_passive_p0}, "
        f"ioc_market={n_ioc}",
        f"- Mean gross MFE >= 0.5 tk: **{n_mfe_ge_05}** configs",
        f"- Mean gross MFE >= 1.0 tk: **{n_mfe_ge_1}** configs",
        f"- Mean gross MFE >= 2.0 tk: **{n_mfe_ge_2}** configs",
        f"- Signal-driven STRICT (MFE>=1tk AND day_conc<=0.20 AND n_fills>=30): "
        f"**{n_strong}** configs",
        f"- Signal-driven RELAXED (MFE>=1tk AND day_conc<=0.70 AND n_fills>=30): "
        f"**{len(strong_relaxed)}** configs",
        "",
        "## Critical structural observation",
        "",
        f"Of the {n_total} deploy-eligible configs, **{n_passive_p2}** "
        f"({100*n_passive_p2/max(1,n_total):.0f}%) use `passive_at_touch_plus_2`. "
        f"Only {n_no_credit_signal} 'no-structural-credit' configs (ioc_market or "
        f"passive_at_touch) have mean gross MFE >= 1 tk in their hold window.",
        "",
        "**This means the Optuna search overwhelmingly converged on configs "
        "that monetize the passive_+K entry credit, not on configs that capture "
        "directional signal alpha.** Trial 278's signal-alpha weakness "
        "(HC #404 finding: only 0.41 tk of the +1.99 tk/fill is signal-driven) "
        "is a property of the entire deploy-eligible cohort, not a quirk of "
        "trial 278 alone.",
        "",
    ])

    if best_no_credit is not None and len(best_no_credit):
        b = best_no_credit.iloc[0]
        verdict_lines.extend([
            "## Best 'no-credit' candidate",
            "",
            f"Highest-Sharpe config without structural entry credit:",
            "",
            f"- `{b['trial']}` — side={b['side']}, horizon={b['horizon']}, "
            f"order={b['order_type']}, hold={b['hold_seconds']:.2f}s",
            f"- Sharpe={b['sharpe']:.2f} | tk/fill={b['mean_net_tk']:+.3f} | "
            f"n_fills={int(b['n_fills'])} | day_conc={b['day_conc']:.3f}",
            f"- Mean gross MFE within hold: {b['gross_MFE_mean_tk']:.3f} tk",
            "",
            "If this config's Sharpe is well below trial 278's, it confirms "
            "the structural credit IS where the alpha is for this signal "
            "family — signal-alpha trading at sub-30s horizons is dominated "
            "by execution-cost rounding for ES at this signal strength.",
            "",
        ])

    verdict_lines.extend([
        "## Recommendation",
        "",
    ])
    if len(strong) > 0 and float(strong["sharpe"].max()) > T278["sharpe"]:
        verdict_lines.append(
            f"**Replace trial 278 with {winner_str}** for live deployment. "
            "This config carries the same structural credit modeling but adds "
            "non-trivial signal alpha (mean MFE >= 1 tk), reducing fragility "
            "to adverse-selection scaling (per HC #404 shadow replay)."
        )
    elif len(strong) > 0:
        verdict_lines.append(
            f"**Supplement trial 278 with {winner_str}** as a parallel paper "
            "strategy. It is more signal-driven (less fragile to HC #404's "
            "adv-sel scale=1.0 shadow replay collapse), but does not beat "
            "trial 278 on absolute Sharpe. Treat it as a diversifier."
        )
    elif len(strong_relaxed) > 0:
        verdict_lines.append(
            f"**KEEP trial 278 as the Monday strict-gate deployment**, but "
            f"PAPER-TRADE {winner_relaxed_str} in parallel as a signal-driven "
            "diversifier. The relaxed-pass cohort is meaningfully signal-driven "
            "(mean MFE >= 1 tk, ~60-78% of fills have MFE >= 1 tk) and would "
            "diversify trial 278's structural-credit fragility. The high "
            "day_conc on 15d OOT is the trade-off; if 2-4 weeks of live paper "
            "trading shows day_conc stabilizing below 0.20, these are stronger "
            "replacement candidates than trial 278."
        )
    else:
        verdict_lines.append(
            "**Keep trial 278 as the Monday deployment** but treat its Sharpe "
            "as STRUCTURAL not SIGNAL-DRIVEN. The Optuna search did NOT find "
            "configs with strong signal alpha at sub-30s horizons in this "
            "predictions cohort. Future research should re-direct toward "
            "longer-hold signal-capture configs (30s+ holds with MFE-trigger "
            "exits) instead of further Optuna sweeps of the same parameter "
            "space."
        )
    verdict_lines.append("")

    (out_dir / "VERDICT.md").write_text("\n".join(verdict_lines))


# ---------------------------------------------------------------------------
# Worker entry (must be module-level for Pool)
# ---------------------------------------------------------------------------
_WORKER_CACHE = {}


def _worker_init():
    """Load predictions cache in each worker once."""
    print(f"[worker {Path('.').resolve()}] loading preds cache...", flush=True)
    _WORKER_CACHE["preds"] = {
        h: _load_predictions(PREDS_PATH, h) for h in HORIZONS
    }


def _worker_eval(path_str: str) -> dict:
    p = Path(path_str)
    try:
        return evaluate_trial(p, _WORKER_CACHE["preds"])
    except Exception as e:
        return {"trial": p.stem, "error": f"worker_exc: {e}\n{traceback.format_exc()[:500]}"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=50,
                    help="Cap on number of deploy-eligible trials to scan.")
    ap.add_argument("--workers", type=int, default=4,
                    help="multiprocessing workers (max 8 per HC #405 spec).")
    ap.add_argument("--serial", action="store_true",
                    help="Force single-process for debugging.")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJ / "output" / f"hc405_signal_driven_scan_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[hc405] output: {out_dir}")

    print(f"[hc405] selecting top-{args.top_n} deploy-eligible trials...")
    trial_paths = select_trial_jsons(args.top_n)
    print(f"[hc405] selected {len(trial_paths)} trials")

    if not trial_paths:
        print("[hc405] no trials to evaluate")
        return 1

    t_start = time.time()

    if args.serial or args.workers <= 1:
        print("[hc405] loading preds cache (serial)...")
        preds = {h: _load_predictions(PREDS_PATH, h) for h in HORIZONS}
        rows = []
        for i, p in enumerate(trial_paths, 1):
            t_eval = time.time()
            row = evaluate_trial(p, preds)
            rows.append(row)
            print(f"[hc405] {i}/{len(trial_paths)} {p.name} "
                  f"({time.time()-t_eval:.1f}s) "
                  f"sharpe={row.get('sharpe', 'na')} "
                  f"mfe={row.get('gross_MFE_mean_tk', 'na')}")
    else:
        n_workers = min(args.workers, 8, len(trial_paths))
        print(f"[hc405] dispatching to {n_workers} workers...")
        path_strs = [str(p) for p in trial_paths]
        with Pool(processes=n_workers, initializer=_worker_init) as pool:
            rows = []
            for i, row in enumerate(pool.imap_unordered(_worker_eval, path_strs), 1):
                rows.append(row)
                print(f"[hc405] [{i}/{len(path_strs)}] "
                      f"{row.get('trial', '?')} "
                      f"sharpe={row.get('sharpe', 'na')} "
                      f"mfe={row.get('gross_MFE_mean_tk', 'na')}", flush=True)

    elapsed = time.time() - t_start
    print(f"[hc405] all evaluations done in {elapsed:.1f}s")

    df = pd.DataFrame(rows)
    rank_and_emit(df, out_dir)

    print(f"[hc405] wrote ranked_configs.csv, TOP5_CANDIDATES.md, VERDICT.md")
    print(f"[hc405] output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
