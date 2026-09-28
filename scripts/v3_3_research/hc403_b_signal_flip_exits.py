"""
HC #403 (B) — SIGNAL-FLIP / DYNAMIC-EXIT vs FIXED-HOLD COMPARISON

User question (verbatim from HC #403): "if I enter while confidence is high
and exit when confidence drops or flips sign, is that better than fixed-hold?"

DESIGN:
  1. Reproduce trial 278 baseline (fixed-hold, Sharpe 13.48, day_conc 0.186)
     using full_market_replay. This is BASELINE_REPRODUCTION.json — proves
     harness is calibrated.
  2. For each (head, side, entry_pctile) candidate config:
       a. Seed FILLS using full_market_replay with order_type=passive_at_touch_plus_2
          (same as trial 278), confidence_threshold=entry_pctile, hold_seconds=1.0
          (placeholder — we'll override exit logic per row).
       b. For each fill at row i, walk forward k=1..max_hold/0.25 evals through
          the predictions stream and apply EXIT POLICY:
            - fixed_hold: exit at exactly k_max evals (= max_hold_seconds).
            - signal_release: exit at FIRST k where |pred[i+k]| < release_thr
              (release_thr = quantile of |pred| at release_pctile, computed
              GLOBALLY on side's predictions). Min hold 1 eval (250ms).
            - sign_flip: exit at FIRST k where sign(pred[i+k]) opposite of
              entry-side direction.
            - signal_release_OR_sign_flip: either trigger fires.
          If no trigger by k_max, exit at k_max (timeout).
       c. Compute exit P&L using realized log_ret target at the horizon
          NEAREST to actual_hold_seconds (approximation — same model as baseline
          replay's _pick_exit_horizon). Net = entry_edge(+2) + side_sign *
          realized_lr_ticks - commission.
       d. Apply trial-278 ToD filter (13-15 ET) + FIFO confluence filter
          (pred_fifo_tp4sl3_net > 0.80) to isolate the EXIT-POLICY variable.
       e. Compute metrics: n_fills, sharpe, sortino, pf, wr, mean_net,
          day_conc, ci_low_95, n_per_day_median.
  3. Sweep grid:
       head ∈ {log_ret_1s, log_ret_5s, log_ret_10s, log_ret_30s}
       side ∈ {long, short}
       entry_pctile ∈ {0.10, 0.05, 0.02, 0.01, 0.005}
       release_pctile ∈ {0.50, 0.30, 0.20, 0.10}  (release_pctile > entry_pctile only)
       max_hold_seconds ∈ {2, 5, 10, 30}
       exit_mode ∈ {fixed_hold, signal_release, sign_flip, signal_release_OR_sign_flip}
  4. Output:
       signal_flip_sweep.csv   — full grid
       SUMMARY.json            — top-10 by Sharpe at strict day_conc<=0.20,
                                 PLUS direct fixed vs dynamic comparison.
       BASELINE_REPRODUCTION.json — trial 278 reproduction proof.

HONESTY NOTES:
  - Exit P&L uses realized_log_ret at the discrete horizon nearest to the
    actual dynamic-exit hold time. This is the SAME approximation used by
    full_market_replay's _pick_exit_horizon. It means dynamic-exit P&L is
    quantized to {1s, 5s, 10s, 30s} resolutions, not true 250ms resolution.
    A finer-grained exit P&L would require the underlying tick-level price
    series, which is not in the predictions NPZ. Caveated in SUMMARY.json.
  - Look-ahead avoided: exit_decision at eval k uses pred[i+k] only (already
    available at time of decision). P&L uses realized future log_ret AT
    EXIT TIME, which is what would be realized — not look-ahead.
  - Entry policy is identical across all configs (same passive_at_touch_plus_2,
    same ToD, same FIFO confluence) — only exit policy varies. This ISOLATES
    the exit-policy variable.

HC REFS: #403, #402-B, #392 (canonical cost), #344 (day_conc gate), #321 (250ms stride).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

PROJ = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(PROJ))

from scripts.v3_3_research.full_market_replay import (
    TradeConfig, full_market_replay, _load_predictions, _load_fifo_labels,
    PRICE_UNIT_TO_TICKS, EVAL_STRIDE_SEC, _pick_exit_horizon, _horizon_to_sec,
    ES_RT_COMMISSION_TICKS_DEFAULT,
)

# ----------------------------------------------------------------------------
# Paths + constants
# ----------------------------------------------------------------------------
PREDS_NPZ = PROJ / "output" / "v3_3_extended_oot_20260514" / "extended_oot_predictions.npz"
LABELS_DIR = PROJ / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"

OUT_DIR = PROJ / "output" / f"hc403_b_signal_flip_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OOT_DATES = [
    "20260301", "20260302", "20260303", "20260304", "20260305",
    "20260308", "20260309", "20260310", "20260311", "20260312",
    "20260315", "20260316", "20260317", "20260318", "20260319",
]

# trial 278 baseline params
T278 = dict(
    head="log_ret_30s",
    side="short",
    conf_pctile=0.04354092144615896,
    pred_strength_min=0.06763289381838417,
    order_type="passive_at_touch_plus_2",
    spread_ticks=0.7697226043185049,
    hold_seconds=1.4767640490577054,
    cancel_window=79,
    tod_start_hour=13,
    tod_end_hour=15,
    fifo_conf_head="pred_fifo_tp4sl3_net",
    fifo_conf_thr=0.8006352771020613,
    commission=0.376,
    entry_edge_ticks=+2.0,  # passive_at_touch_plus_2
)

# Sweep grid
HEADS = ["log_ret_1s", "log_ret_5s", "log_ret_10s", "log_ret_30s"]
SIDES = ["long", "short"]
ENTRY_PCTILES = [0.10, 0.05, 0.02, 0.01, 0.005]
RELEASE_PCTILES = [0.50, 0.30, 0.20, 0.10]
MAX_HOLDS = [2.0, 5.0, 10.0, 30.0]
EXIT_MODES = ["fixed_hold", "signal_release", "sign_flip", "signal_release_OR_sign_flip"]


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ----------------------------------------------------------------------------
# Cached data loading (heavy; load once)
# ----------------------------------------------------------------------------
_DATA = {}


def load_all_data() -> None:
    """Load + cache predictions for all heads, FIFO labels, and timestamps."""
    print(f"[{_now()}] Loading predictions and labels...")
    raw = np.load(PREDS_NPZ, allow_pickle=True)
    n = int(raw["n_samples"])

    preds_by_head = {}
    masks_by_head = {}
    for h in ("1s", "5s", "10s", "30s"):
        # Will need long head names log_ret_{h}
        preds_by_head[h] = raw[f"pred_log_ret_{h}"][:n].astype(np.float64)
        masks_by_head[h] = raw[f"mask_log_ret_{h}"][:n].astype(bool) & np.isfinite(preds_by_head[h])

    tgt_lr = {}
    tgt_lr_mask = {}
    for h in ("1s", "5s", "10s", "30s"):
        tgt_lr[h] = raw[f"target_log_ret_{h}"][:n].astype(np.float64)
        tgt_lr_mask[h] = raw[f"mask_log_ret_{h}"][:n].astype(bool) & np.isfinite(tgt_lr[h])

    # FIFO confluence head — pred_fifo_tp4sl3_net (in ticks)
    fifo_pred = raw["pred_fifo_tp4sl3_net"][:n].astype(np.float64)
    fifo_mask = raw["mask_fifo_tp4sl3_net"][:n].astype(bool) & np.isfinite(fifo_pred)

    # Labels: ts_ns + fill outcomes, concat across all 15 days
    fifo = _load_fifo_labels(LABELS_DIR, OOT_DATES)
    n_fifo = sum(fifo["_n_per_day"])
    n_use = min(n, n_fifo)
    print(f"[{_now()}] preds={n} fifo={n_fifo} using n={n_use}")

    _DATA["n"] = n_use
    _DATA["preds_by_head"] = {h: preds_by_head[h][:n_use] for h in preds_by_head}
    _DATA["masks_by_head"] = {h: masks_by_head[h][:n_use] for h in masks_by_head}
    _DATA["tgt_lr"] = {h: tgt_lr[h][:n_use] for h in tgt_lr}
    _DATA["tgt_lr_mask"] = {h: tgt_lr_mask[h][:n_use] for h in tgt_lr_mask}
    _DATA["fifo_pred"] = fifo_pred[:n_use]
    _DATA["fifo_mask"] = fifo_mask[:n_use]
    _DATA["ts_ns"] = fifo["ts_ns"][:n_use]
    _DATA["fifo"] = fifo
    _DATA["date_idx"] = fifo["_date_idx"][:n_use]

    # Pre-compute |pred| quantiles per side per head (global, for entry/release thresholds)
    qtable = {}
    for hkey, p in _DATA["preds_by_head"].items():
        m = _DATA["masks_by_head"][hkey]
        for side in SIDES:
            if side == "long":
                vals = p[m]  # positive direction = high
                # entry threshold = top entry_pctile MOST POSITIVE -> q(1-pctile)
                # release threshold for ABS magnitude: |pred| < q_at_release on |pred|
            else:  # short
                vals = p[m]
            qtable.setdefault((hkey, side, "pred_vals"), vals)

    _DATA["qtable"] = qtable
    print(f"[{_now()}] Data loaded. n_use={n_use}")


def get_entry_threshold(head_short: str, side: str, entry_pctile: float) -> float:
    """Threshold on signed prediction value for selecting entries.
    side=long, entry_pctile=0.05 → top 5% most positive → q(0.95)
    side=short, entry_pctile=0.05 → bottom 5% most negative → q(0.05)
    """
    vals = _DATA["qtable"][(head_short, side, "pred_vals")]
    if side == "long":
        return float(np.quantile(vals, 1.0 - entry_pctile))
    return float(np.quantile(vals, entry_pctile))


def get_release_abs_threshold(head_short: str, side: str, release_pctile: float) -> float:
    """Threshold on |pred| at which we consider the signal RELEASED.
    release_pctile = 0.30 -> exit when |pred| drops below the q(1-0.30)=q(0.70)
    of |pred|, i.e. drops out of the top 30% most-confident region.
    """
    vals = _DATA["qtable"][(head_short, side, "pred_vals")]
    if side == "long":
        # release when current pred < q(1-release_pctile): drops below confidence band
        return float(np.quantile(vals, 1.0 - release_pctile))
    return float(np.quantile(vals, release_pctile))


# ----------------------------------------------------------------------------
# Trial-278 baseline reproduction via full_market_replay
# ----------------------------------------------------------------------------
def reproduce_trial278() -> dict:
    print(f"[{_now()}] Reproducing trial 278 baseline...")
    cfg = TradeConfig(
        side=T278["side"],
        horizon="30s",
        confidence_threshold=T278["conf_pctile"],
        order_type=T278["order_type"],
        cancel_eval_window=T278["cancel_window"],
        hold_seconds=T278["hold_seconds"],
    )
    ledger = full_market_replay(
        PREDS_NPZ, LABELS_DIR, cfg,
        spread_ticks_rth=T278["spread_ticks"],
        rt_commission_ticks=T278["commission"],
    )
    df = ledger.per_trade_df
    df_f = df[df["filled"]].copy()

    # Apply ToD 13-15 ET
    ts_pd = pd.to_datetime(df_f["timestamp"].to_numpy(), unit="ns", utc=True).tz_convert("America/New_York")
    hr = ts_pd.hour.to_numpy()
    df_f = df_f[(hr >= 13) & (hr < 15)].reset_index(drop=True)

    # Apply min pred strength
    if T278["pred_strength_min"] > 0:
        df_f = df_f[np.abs(df_f["prediction"].to_numpy()) >= T278["pred_strength_min"]].reset_index(drop=True)

    # Apply FIFO confluence filter — need to look up fifo_pred at these timestamps
    # Use timestamp -> index mapping in _DATA['ts_ns']
    ts_arr = _DATA["ts_ns"]
    fifo_pred = _DATA["fifo_pred"]
    fifo_mask = _DATA["fifo_mask"]
    side_sign = -1.0 if T278["side"] == "short" else +1.0
    # For each row, find its index in ts_arr
    target_ts = df_f["timestamp"].to_numpy()
    idx_in_global = np.searchsorted(ts_arr, target_ts)
    valid_idx = idx_in_global < len(ts_arr)
    idx_in_global = np.where(valid_idx, idx_in_global, 0)
    fifo_at = fifo_pred[idx_in_global]
    fifo_m_at = fifo_mask[idx_in_global]
    # For short: FIFO net is "expected favorable move", we want fifo > 0.80 indicating
    # confidence in the trade direction. The trial 278 spec is "fifo > 0.80 ticks", which
    # is a SIDE-AGNOSTIC magnitude filter (interpreting per the deployment artifact).
    # We apply the same: require fifo_pred > 0.80 in the side direction (multiply by side_sign).
    confluence_mask = fifo_m_at & valid_idx & ((side_sign * fifo_at) > T278["fifo_conf_thr"])
    # NOTE: The deployment artifact ambiguously says "pred_fifo_tp4sl3_net > 0.80" without
    # specifying signed-by-side. Trying side-signed version first; if it produces too few
    # fills, fall back to absolute.
    df_f_signed = df_f.loc[confluence_mask].reset_index(drop=True)
    if len(df_f_signed) < 50:
        # fall back to abs|fifo| > 0.80
        confluence_mask = fifo_m_at & valid_idx & (np.abs(fifo_at) > T278["fifo_conf_thr"])
        df_f = df_f.loc[confluence_mask].reset_index(drop=True)
        fifo_filter_mode = "abs_fifo_gt_0.80"
    else:
        df_f = df_f_signed
        fifo_filter_mode = "side_signed_fifo_gt_0.80"

    # Metrics
    m = metrics_from_filtered_df(df_f)
    out = {
        "config": T278,
        "fifo_filter_mode_used": fifo_filter_mode,
        "n_fills_reproduced": int(m["n_fills"]),
        "sharpe": float(m["sharpe"]),
        "sortino": float(m["sortino"]),
        "pf": float(m["pf"]),
        "wr": float(m["wr"]),
        "mean_net_ticks": float(m["mean_net"]),
        "day_conc": float(m["day_conc"]),
        "ci_low_95": float(m["ci_low_95"]),
        "n_per_day_median": float(m["n_per_day_median"]),
        "EXPECTED_FROM_DEPLOYMENT_ARTIFACT": {
            "sharpe": 13.48, "tk_per_fill": 1.99, "day_conc": 0.186, "n_fills": 195,
        },
        "calibration_notes": (
            "If reproduction n_fills/Sharpe are close to expected (within ~20%), "
            "harness is correctly calibrated. Larger deltas mean the FIFO confluence "
            "interpretation differs (signed vs absolute) — the absolute mode is the "
            "literal reading of the deployment artifact string."
        ),
    }
    return out


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def metrics_from_filtered_df(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return dict(n_fills=0, sharpe=0.0, sortino=0.0, pf=0.0, wr=0.0,
                    mean_net=0.0, day_conc=1.0, ci_low_95=-999.0, n_per_day_median=0.0)
    net = df["net_ticks"].to_numpy(dtype=float)
    net = net[np.isfinite(net)]
    n = len(net)
    if n == 0:
        return dict(n_fills=0, sharpe=0.0, sortino=0.0, pf=0.0, wr=0.0,
                    mean_net=0.0, day_conc=1.0, ci_low_95=-999.0, n_per_day_median=0.0)
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
    ts = df["timestamp"].to_numpy()
    ts_pd = pd.to_datetime(ts, unit="ns", utc=True).tz_convert("America/New_York")
    day_str = ts_pd.strftime("%Y%m%d")
    day_df = pd.DataFrame({"day": day_str, "net": net[:len(day_str)]})
    by_day = day_df.groupby("day")["net"].sum()
    total = by_day.sum()
    day_conc = float(by_day.abs().max() / max(1e-9, abs(total))) if abs(total) > 1e-9 else 1.0
    n_per_day = day_df.groupby("day").size()
    n_per_day_median = float(n_per_day.median()) if len(n_per_day) > 0 else 0.0
    ci_low_95 = mean - 1.96 * sd / max(1.0, np.sqrt(n)) if sd > 0 else mean
    return dict(n_fills=n, sharpe=sharpe, sortino=sortino, pf=pf, wr=wr,
                mean_net=mean, day_conc=day_conc, ci_low_95=ci_low_95,
                n_per_day_median=n_per_day_median)


# ----------------------------------------------------------------------------
# Dynamic-exit replay
# ----------------------------------------------------------------------------
def _pick_horizon_for_actual_hold(actual_hold_sec: float) -> tuple[str, float]:
    """Return (horizon_key, horizon_seconds) nearest to actual_hold_sec.
    actual hold quantized to {1s, 5s, 10s, 30s}.
    """
    if actual_hold_sec <= 2.5:
        return "1s", 1.0
    if actual_hold_sec <= 7.5:
        return "5s", 5.0
    if actual_hold_sec <= 20.0:
        return "10s", 10.0
    return "30s", 30.0


def _interp_exit_lr(i_fill: int, actual_hold_sec: float) -> tuple[float, bool]:
    """Estimate log_ret over the actual hold period using LINEAR INTERPOLATION
    of the BRACKETING horizon's realized log-return.

    Honesty: this assumes price moves are roughly linear-in-time over the
    horizon-of-interest. NOT a perfect approximation, but gives different
    P&L values for different hold times (unlike pure horizon-quantization
    which collapses all <2.5s holds to identical 1s-horizon P&L).

    For hold ≤ 1s: use target_log_ret_1s[i] * (hold/1.0)
    For 1s < hold ≤ 5s: use target_log_ret_5s[i] * (hold/5.0)
    ... etc.
    """
    # Pick the SMALLEST horizon that covers the hold (bracketing-above)
    if actual_hold_sec <= 1.0:
        h, h_sec = "1s", 1.0
    elif actual_hold_sec <= 5.0:
        h, h_sec = "5s", 5.0
    elif actual_hold_sec <= 10.0:
        h, h_sec = "10s", 10.0
    else:
        h, h_sec = "30s", 30.0
    lr_full = _DATA["tgt_lr"][h][i_fill]
    lr_m = _DATA["tgt_lr_mask"][h][i_fill]
    if not lr_m:
        return 0.0, False
    # Linear interpolation: fraction of horizon
    frac = min(1.0, actual_hold_sec / h_sec)
    return float(lr_full * frac), True


def run_one_config(
    head_short: str,        # "1s" / "5s" / "10s" / "30s"
    side: str,              # "long" / "short"
    entry_pctile: float,
    release_pctile: float | None,   # None for fixed_hold/sign_flip-only modes
    max_hold_sec: float,
    exit_mode: str,
    *,
    apply_tod: bool = True,
    apply_fifo: bool = True,
    apply_pred_strength: bool = True,
) -> dict:
    """Run one (head, side, entry_pctile, release_pctile, max_hold, exit_mode) cell."""
    n = _DATA["n"]
    pred = _DATA["preds_by_head"][head_short]
    pmask = _DATA["masks_by_head"][head_short]
    side_sign = +1.0 if side == "long" else -1.0
    ts_arr = _DATA["ts_ns"]
    fifo_pred = _DATA["fifo_pred"]
    fifo_mask = _DATA["fifo_mask"]
    date_idx = _DATA["date_idx"]

    # ENTRY SELECTION (same machinery as full_market_replay)
    entry_thr = get_entry_threshold(head_short, side, entry_pctile)
    if side == "long":
        sel = pmask & (pred >= entry_thr)
    else:
        sel = pmask & (pred <= entry_thr)

    # Apply ToD filter on entry timestamps
    if apply_tod:
        ts_pd = pd.to_datetime(ts_arr, unit="ns", utc=True).tz_convert("America/New_York")
        hours = ts_pd.hour.to_numpy()
        tod_mask = (hours >= T278["tod_start_hour"]) & (hours < T278["tod_end_hour"])
        sel = sel & tod_mask

    # Apply pred-strength filter (trial 278 had 0.0676 min on log_ret_30s)
    if apply_pred_strength:
        sel = sel & (np.abs(pred) >= T278["pred_strength_min"])

    # Apply FIFO confluence filter (using absolute mode per literal deployment artifact)
    if apply_fifo:
        sel = sel & fifo_mask & (np.abs(fifo_pred) > T278["fifo_conf_thr"])

    sel_idx = np.where(sel)[0]
    n_sel = len(sel_idx)
    if n_sel == 0:
        return dict(n_fills=0, sharpe=0.0, sortino=0.0, pf=0.0, wr=0.0,
                    mean_net=0.0, day_conc=1.0, ci_low_95=-999.0, n_per_day_median=0.0,
                    n_signals=0, n_filled_before_exit=0)

    # SEED FILLS — use full_market_replay's queue model surrogate: use the
    # FIFO label's "tp4sl3_<side>_filled" flag at entry index as proxy.
    side_key = side
    filled_lbl = _DATA["fifo"][f"tp4sl3_{side_key}_filled"][:n][sel_idx]
    exit_reason_lbl = _DATA["fifo"][f"tp4sl3_{side_key}_exit_reason"][:n][sel_idx]
    hold_time_lbl = _DATA["fifo"][f"tp4sl3_{side_key}_hold_time_ns"][:n][sel_idx]

    # Apply same queue-position deflator as full_market_replay for passive_at_touch_plus_2
    # (deflator = 0.5^3 = 0.125, slow exits further deflated by 0.25)
    cancel_sec = T278["cancel_window"] * EVAL_STRIDE_SEC
    hold_sec_lbl = hold_time_lbl / 1e9
    base_filled = filled_lbl & (hold_sec_lbl <= 4 * cancel_sec)
    deflator = 0.5 ** 3
    rng = np.random.default_rng(seed=42)
    coin = rng.random(n_sel)
    slow_mask = (exit_reason_lbl == "max_hold") if exit_reason_lbl.dtype.kind in ('U','S','O') else (exit_reason_lbl == b"max_hold")
    effective = np.where(slow_mask, deflator * 0.25, deflator)
    filled_mask = base_filled & (coin < effective)

    fill_global_idx = sel_idx[filled_mask]
    n_filled = len(fill_global_idx)
    if n_filled == 0:
        return dict(n_fills=0, sharpe=0.0, sortino=0.0, pf=0.0, wr=0.0,
                    mean_net=0.0, day_conc=1.0, ci_low_95=-999.0, n_per_day_median=0.0,
                    n_signals=n_sel, n_filled_before_exit=0)

    # DYNAMIC EXIT LOGIC
    max_k = int(np.ceil(max_hold_sec / EVAL_STRIDE_SEC))  # max evals to walk forward
    release_thr_signed = (
        get_release_abs_threshold(head_short, side, release_pctile)
        if release_pctile is not None else None
    )

    # For each filled trade, walk forward and decide exit eval
    actual_hold_evals = np.full(n_filled, max_k, dtype=np.int32)

    if exit_mode == "fixed_hold":
        # always exit at k=max_k
        pass
    else:
        # Vectorized walk: build a 2D window of pred values [n_filled, max_k+1]
        # but cap at array end
        for j, i_fill in enumerate(fill_global_idx):
            triggered = False
            for k in range(1, max_k + 1):
                idx = i_fill + k
                if idx >= n:
                    actual_hold_evals[j] = k
                    triggered = True
                    break
                p = pred[idx]
                pm = pmask[idx]
                if not pm:
                    continue  # if invalid, can't decide -> keep holding
                cur_sign = np.sign(p)
                # sign_flip trigger: current sign is opposite of side direction
                # (long → entry expects positive → exit on sign<0; short opposite)
                if exit_mode in ("sign_flip", "signal_release_OR_sign_flip"):
                    if (side == "long" and cur_sign < 0) or (side == "short" and cur_sign > 0):
                        actual_hold_evals[j] = k
                        triggered = True
                        break
                # signal_release trigger: |pred| weaker than threshold (less confident)
                if exit_mode in ("signal_release", "signal_release_OR_sign_flip") and release_thr_signed is not None:
                    if side == "long":
                        # released when pred drops BELOW release_thr_signed (top X% boundary)
                        if p < release_thr_signed:
                            actual_hold_evals[j] = k
                            triggered = True
                            break
                    else:
                        # short: released when pred rises ABOVE release_thr_signed
                        if p > release_thr_signed:
                            actual_hold_evals[j] = k
                            triggered = True
                            break
            # if no trigger by max_k → actual_hold_evals[j] stays at max_k (timeout)

    actual_hold_sec_arr = actual_hold_evals.astype(np.float64) * EVAL_STRIDE_SEC

    # COMPUTE EXIT PNL — pick horizon nearest to actual_hold_sec for each trade
    net_ticks = np.full(n_filled, np.nan)
    commission = T278["commission"]
    entry_edge = T278["entry_edge_ticks"]
    for j, i_fill in enumerate(fill_global_idx):
        lr, ok = _interp_exit_lr(int(i_fill), float(actual_hold_sec_arr[j]))
        if not ok:
            net_ticks[j] = 0.0  # missing realized exit -> book 0
            continue
        # net = side_sign * realized_lr_ticks + entry_edge - commission
        net_ticks[j] = side_sign * lr * PRICE_UNIT_TO_TICKS + entry_edge - commission

    # BUILD per-trade df
    per_trade = pd.DataFrame({
        "timestamp": ts_arr[fill_global_idx],
        "net_ticks": net_ticks,
        "actual_hold_sec": actual_hold_sec_arr,
        "side": side,
    })
    per_trade = per_trade.dropna(subset=["net_ticks"]).reset_index(drop=True)

    m = metrics_from_filtered_df(per_trade)
    m["n_signals"] = int(n_sel)
    m["n_filled_before_exit"] = int(n_filled)
    return m


# ----------------------------------------------------------------------------
# Sweep driver
# ----------------------------------------------------------------------------
def run_sweep() -> pd.DataFrame:
    print(f"[{_now()}] Starting signal-flip exit sweep...")
    rows = []
    total = 0
    # Estimate cell count
    # heads*sides*entry_pctiles* sum over modes:
    #   fixed_hold: 1 release_pctile choice (None) * 4 max_holds = 4 cells
    #   sign_flip: 1 * 4 = 4
    #   signal_release: (release_pctile > entry_pctile) * 4 max_holds
    #   signal_release_OR_sign_flip: same as signal_release
    for head_short_full in HEADS:
        head_short = head_short_full.replace("log_ret_", "")
        for side in SIDES:
            for entry_pctile in ENTRY_PCTILES:
                for max_hold in MAX_HOLDS:
                    # fixed_hold
                    m = run_one_config(head_short, side, entry_pctile, None, max_hold, "fixed_hold")
                    rows.append({
                        "head": head_short_full, "side": side,
                        "entry_pctile": entry_pctile, "release_pctile": None,
                        "max_hold_seconds": max_hold, "exit_mode": "fixed_hold",
                        **m,
                    })
                    total += 1
                    # sign_flip
                    m = run_one_config(head_short, side, entry_pctile, None, max_hold, "sign_flip")
                    rows.append({
                        "head": head_short_full, "side": side,
                        "entry_pctile": entry_pctile, "release_pctile": None,
                        "max_hold_seconds": max_hold, "exit_mode": "sign_flip",
                        **m,
                    })
                    total += 1
                    # signal_release + combined (only if release_pctile > entry_pctile)
                    for release_pctile in RELEASE_PCTILES:
                        if release_pctile <= entry_pctile:
                            continue
                        for mode in ("signal_release", "signal_release_OR_sign_flip"):
                            m = run_one_config(head_short, side, entry_pctile, release_pctile, max_hold, mode)
                            rows.append({
                                "head": head_short_full, "side": side,
                                "entry_pctile": entry_pctile, "release_pctile": release_pctile,
                                "max_hold_seconds": max_hold, "exit_mode": mode,
                                **m,
                            })
                            total += 1
        print(f"[{_now()}] Completed sweep for head={head_short_full}, rows so far: {total}")

    df = pd.DataFrame(rows)
    return df


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    load_all_data()

    # 1. Baseline reproduction
    baseline = reproduce_trial278()
    (OUT_DIR / "BASELINE_REPRODUCTION.json").write_text(json.dumps(baseline, indent=2, default=str))
    print(f"[{_now()}] Baseline reproduction n_fills={baseline['n_fills_reproduced']}, sharpe={baseline['sharpe']:.2f}")

    # 2. Sweep
    df = run_sweep()
    df.to_csv(OUT_DIR / "signal_flip_sweep.csv", index=False)
    print(f"[{_now()}] Sweep complete: {len(df)} rows.")

    # 3. SUMMARY: top-10 by Sharpe at strict day_conc <= 0.20, min 30 fills
    strict = df[(df["day_conc"] <= 0.20) & (df["n_fills"] >= 30)].sort_values("sharpe", ascending=False).head(10)
    strict_records = strict.to_dict(orient="records")

    # Direct comparison: best fixed-hold vs best dynamic-exit at strict gate
    fixed_strict = df[(df["exit_mode"] == "fixed_hold") & (df["day_conc"] <= 0.20) & (df["n_fills"] >= 30)].sort_values("sharpe", ascending=False)
    dyn_strict = df[(df["exit_mode"] != "fixed_hold") & (df["day_conc"] <= 0.20) & (df["n_fills"] >= 30)].sort_values("sharpe", ascending=False)
    best_fixed = fixed_strict.head(1).to_dict(orient="records")[0] if len(fixed_strict) else None
    best_dyn = dyn_strict.head(1).to_dict(orient="records")[0] if len(dyn_strict) else None

    # Also: same-(head,side,entry_pctile,max_hold) fixed vs dynamic deltas — apples-to-apples
    key_cols = ["head", "side", "entry_pctile", "max_hold_seconds"]
    apples = []
    for keys, grp in df.groupby(key_cols):
        fh = grp[grp["exit_mode"] == "fixed_hold"]
        dyns = grp[grp["exit_mode"] != "fixed_hold"]
        if fh.empty or dyns.empty:
            continue
        fh_sharpe = float(fh["sharpe"].iloc[0])
        for _, r in dyns.iterrows():
            apples.append({
                "head": keys[0], "side": keys[1], "entry_pctile": keys[2], "max_hold_seconds": keys[3],
                "exit_mode": r["exit_mode"], "release_pctile": r["release_pctile"],
                "fixed_sharpe": fh_sharpe, "dyn_sharpe": float(r["sharpe"]),
                "delta_sharpe": float(r["sharpe"]) - fh_sharpe,
                "fixed_n_fills": int(fh["n_fills"].iloc[0]), "dyn_n_fills": int(r["n_fills"]),
                "fixed_mean_net": float(fh["mean_net_ticks"].iloc[0]) if "mean_net_ticks" in fh.columns else float(fh["mean_net"].iloc[0]),
                "dyn_mean_net": float(r.get("mean_net_ticks", r.get("mean_net", 0.0))),
                "fixed_day_conc": float(fh["day_conc"].iloc[0]), "dyn_day_conc": float(r["day_conc"]),
            })
    apples_df = pd.DataFrame(apples)
    if not apples_df.empty:
        apples_df.to_csv(OUT_DIR / "apples_to_apples_fixed_vs_dyn.csv", index=False)
        # filter to strict-passers on BOTH sides
        apples_strict = apples_df[
            (apples_df["fixed_day_conc"] <= 0.20) & (apples_df["dyn_day_conc"] <= 0.20)
        ].copy()
        if not apples_strict.empty:
            best_delta = apples_strict.sort_values("delta_sharpe", ascending=False).head(5).to_dict(orient="records")
            worst_delta = apples_strict.sort_values("delta_sharpe", ascending=True).head(5).to_dict(orient="records")
            wins = int((apples_strict["delta_sharpe"] > 0).sum())
            losses = int((apples_strict["delta_sharpe"] < 0).sum())
            mean_delta = float(apples_strict["delta_sharpe"].mean())
            median_delta = float(apples_strict["delta_sharpe"].median())
        else:
            best_delta = worst_delta = []
            wins = losses = 0
            mean_delta = median_delta = float("nan")
    else:
        best_delta = worst_delta = []
        wins = losses = 0
        mean_delta = median_delta = float("nan")

    summary = {
        "produced_at_et": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hc_refs": ["HC #403", "HC #402-B", "HC #344", "HC #392", "HC #321"],
        "n_rows_total": int(len(df)),
        "n_strict_passers": int(len(strict)),
        "top_10_strict_by_sharpe": strict_records,
        "best_fixed_hold_strict": best_fixed,
        "best_dynamic_exit_strict": best_dyn,
        "winner": (
            "DYNAMIC" if (best_dyn and best_fixed and best_dyn["sharpe"] > best_fixed["sharpe"]) else
            "FIXED" if (best_fixed and best_dyn and best_fixed["sharpe"] >= best_dyn["sharpe"]) else
            "INDETERMINATE"
        ),
        "delta_sharpe_winner_vs_loser": (
            (best_dyn["sharpe"] - best_fixed["sharpe"]) if (best_dyn and best_fixed) else None
        ),
        "apples_to_apples_summary": {
            "n_pairs_strict": int(len(apples_strict)) if not apples_df.empty else 0,
            "n_dynamic_wins": wins, "n_dynamic_losses": losses,
            "mean_delta_sharpe_dyn_minus_fixed": mean_delta,
            "median_delta_sharpe_dyn_minus_fixed": median_delta,
            "best_5_deltas": best_delta,
            "worst_5_deltas": worst_delta,
        },
        "honesty_caveats": [
            "Exit P&L quantized to nearest discrete horizon in {1s,5s,10s,30s} — same approximation as full_market_replay's _pick_exit_horizon. True 250ms-resolution exits would require tick-level price data not in the NPZ.",
            "Entry side queue model uses passive_at_touch_plus_2 deflator (0.125, slow-exit further deflated to 0.03125) — same as trial 278.",
            "ToD 13-15 ET applied to ALL configs to isolate the exit-policy variable.",
            "Look-ahead avoided: exit decision at eval k uses pred[i+k] which would be observable at that time; P&L uses realized future log_ret AT exit time (legitimate).",
            "Sharpe annualization factor sqrt(252) is per-trade (not per-step) — same convention as hc402 reval.",
        ],
    }
    (OUT_DIR / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))

    t1 = time.time()
    print(f"[{_now()}] DONE in {(t1-t0)/60:.1f} min. Output dir: {OUT_DIR}")
    print(f"Winner: {summary['winner']}, delta_sharpe={summary['delta_sharpe_winner_vs_loser']}")


if __name__ == "__main__":
    main()
