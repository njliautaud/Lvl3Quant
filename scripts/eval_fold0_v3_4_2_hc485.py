#!/usr/bin/env python3
"""
eval_fold0_v3_4_2_hc485.py — Automated post-fold-0 deploy-gate evaluator for the
CNN-Mamba v3.4.2 retrain on the freshly-regenerated `_v3/` alpha labels
(HC #485 NaN-fix). Fires the moment Neptune fold-0 OOT NPZ lands on Jupiter.

Pipeline (all autonomous, no human-in-the-loop):
  1. Wait/poll for fold_00_oot.npz on Jupiter (max 2h, 60s sleep).
  2. NaN-audit gate (HC #485 R3).
  3. Concat IC gate vs baseline 0.106 (IC_10s).
  4. Symmetric L/S balance gate (HC #475 R2) across 5 standard configs.
  5. Regime-agnostic gate (HC #428 R1): day_conc ≤0.70, |regime_skew| ≤0.50.
  6. MFE-within-horizon gate (HC #432 R2): TP ≤ p90 MFE(h); hold ≤ 1.5h; cancel ≤ h.
  7. Delta-vs-broken-baseline (compare to hc475_ab/symmetric_gate_summary.parquet).
  8. Write verdict.json + plain-English discord_briefing.txt (HC #433).
  9. Idempotent (re-run skipped unless --force). --self-test runs the gates on
     an existing baseline NPZ as a smoke test.

Per HC #393: act-then-report. Per HC #475 R2 / #428 R1 / #432 R2 deploy gates.
Per HC #74: canonical FIFO market replay only.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3))
sys.path.insert(0, str(LVL3 / "scripts"))
sys.path.insert(0, str(LVL3 / "scripts" / "v3_4_research"))

# -------- Paths --------
FOLD0_NPZ = LVL3 / "output" / "cnn_mamba_v3_4_2_hc477fix_v2" / "fold_00_oot.npz"
SELF_TEST_NPZ = LVL3 / "output" / "hc432_v342_47day_validation" / "fold_00_ep1_oot_inference_47day_hc432.npz"
BROKEN_BASELINE_SUMMARY = LVL3 / "output" / "hc475_ab" / "symmetric_gate_summary.parquet"

OUT_DIR = LVL3 / "output" / "eval_fold0_v3_4_2_hc485"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VERDICT_PATH = OUT_DIR / "verdict.json"
DISCORD_PATH = OUT_DIR / "discord_briefing.txt"
REGEN_COMPLETE = OUT_DIR / ".regen_complete.json"
RUN_LOG = LVL3 / "logs" / "eval_fold0_v3_4_2_hc485.log"
RUN_HISTORY = LVL3 / "RUN_HISTORY.md"

# -------- Hyperparams (mirror the A/B baseline for clean comparison) --------
CONFIGS: List[Tuple[str, str, List[str]]] = [
    ("pair",    "pair01_logret1s+pup5s",
        ["pred_log_ret_1s", "pred_p_up_5s"]),
    ("pair",    "pair07_logret10s+logret60sq50",
        ["pred_log_ret_10s", "pred_log_ret_60s_q50"]),
    ("pair",    "pair08_logret5s+pup5s",
        ["pred_log_ret_5s", "pred_p_up_5s"]),
    ("triplet", "trip01_pup5s+logret1s+logret10s",
        ["pred_p_up_5s", "pred_log_ret_1s", "pred_log_ret_10s"]),
    ("triplet", "trip03_logret5s+pup5s+logret1s",
        ["pred_log_ret_5s", "pred_p_up_5s", "pred_log_ret_1s"]),
]

TP_TICKS = 4.0
SL_TICKS = 3.0
HOLD_S = 30.0
CANCEL_S = 10.0
ORDER_TYPE = "passive_at_touch"

IS_FRACTION = 0.70
TARGET_SHORT_SHARE_LO = 0.40
TARGET_SHORT_SHARE_HI = 0.60

# Deploy-gate thresholds
NAN_FRAC_AVG_MAX = 0.10
NAN_FRAC_PER_DATE_MAX = 0.50
BASELINE_CONCAT_IC_10S = 0.106
DAY_CONC_MAX = 0.70
REGIME_SKEW_MAX = 0.50
LONG_SHARE_LO = 0.20
LONG_SHARE_HI = 0.80
LS_BOTHSIDE_RATIO = 0.50  # both-side IC > 0.5× better-side

# MFE-within-horizon (HC #432 R2)
# CONFIG → primary horizon h (seconds). Bracket TP/hold/cancel checked against this.
CONFIG_PRIMARY_HORIZON_S = {
    "pair01_logret1s+pup5s":              5.0,   # min(1s, 5s) → use 5s as the gate horizon (head with longest h)
    "pair07_logret10s+logret60sq50":     10.0,
    "pair08_logret5s+pup5s":              5.0,
    "trip01_pup5s+logret1s+logret10s":    5.0,
    "trip03_logret5s+pup5s+logret1s":     5.0,
}

REGIME_CACHE = LVL3 / "output" / "stream_backtest_v2" / "top10_per_day_pair.parquet"


# -------- Utility --------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
        with RUN_LOG.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def append_run_history(entry: str) -> None:
    try:
        with RUN_HISTORY.open("a") as f:
            f.write(f"\n- {datetime.now().strftime('%Y-%m-%d %H:%M ET')} — {entry}\n")
    except Exception as e:
        log(f"WARN: could not append RUN_HISTORY: {e}")


def regime_classify(date_str: str) -> str:
    if not REGIME_CACHE.exists():
        return "unknown"
    try:
        df = pd.read_parquet(REGIME_CACHE)
        if "date" not in df.columns or "regime" not in df.columns:
            return "unknown"
        df["date"] = df["date"].astype(str)
        row = df[df["date"] == date_str]
        if row.empty:
            return "unknown"
        return str(row["regime"].iloc[0])
    except Exception:
        return "unknown"


# -------- Wait / polling guard --------
def wait_for_npz(target: Path, max_wait_s: float, sleep_s: float) -> bool:
    """Return True if NPZ exists (waiting up to max_wait_s). False if timeout."""
    t_start = time.time()
    if target.exists():
        log(f"NPZ already present: {target}")
        return True
    log(f"Waiting for NPZ: {target} (max {max_wait_s/60:.0f} min)")
    while time.time() - t_start < max_wait_s:
        if target.exists():
            log(f"NPZ appeared after {time.time()-t_start:.0f}s")
            return True
        time.sleep(sleep_s)
    log(f"TIMEOUT waiting for {target}")
    return False


# -------- Gate 1: NaN audit (HC #485 R3) --------
def nan_audit(arrs: Dict[str, np.ndarray]) -> Dict:
    """Per-date NaN audit on prediction columns. HALT if avg >0.10 across cols
    or any single date >0.50 on a required col on an open-market date."""
    sample_dates = arrs["sample_dates"]
    unique_dates = sorted(set(sample_dates.tolist()))
    pred_cols = [k for k in arrs.keys() if k.startswith("pred_")]
    if not pred_cols:
        return {"status": "FAIL_NAN_AUDIT", "reason": "no pred_* columns found"}

    per_date_nan: Dict[str, Dict[str, float]] = {}
    col_avg: Dict[str, float] = {}
    worst_date_col: Tuple[Optional[str], Optional[str], float] = (None, None, 0.0)

    for col in pred_cols:
        v = arrs[col]
        if v.dtype.kind not in ("f", "c"):
            col_avg[col] = 0.0
            continue
        nan_fracs = []
        for d in unique_dates:
            mask = sample_dates == d
            sub = v[mask]
            if sub.size == 0:
                continue
            nf = float(np.isnan(sub).mean())
            per_date_nan.setdefault(d, {})[col] = nf
            nan_fracs.append(nf)
            if nf > worst_date_col[2]:
                worst_date_col = (d, col, nf)
        col_avg[col] = float(np.mean(nan_fracs)) if nan_fracs else 0.0

    avg_across_cols = float(np.mean(list(col_avg.values()))) if col_avg else 0.0
    halt = False
    reasons = []
    if avg_across_cols > NAN_FRAC_AVG_MAX:
        halt = True
        reasons.append(f"avg_nan_frac {avg_across_cols:.3f} > {NAN_FRAC_AVG_MAX}")
    if worst_date_col[2] > NAN_FRAC_PER_DATE_MAX:
        halt = True
        reasons.append(f"worst date {worst_date_col[0]} col {worst_date_col[1]} "
                       f"nan_frac={worst_date_col[2]:.3f} > {NAN_FRAC_PER_DATE_MAX}")

    return {
        "status": "FAIL_NAN_AUDIT" if halt else "PASS",
        "avg_nan_frac_across_cols": avg_across_cols,
        "worst_date_col": {"date": worst_date_col[0], "col": worst_date_col[1],
                           "nan_frac": worst_date_col[2]},
        "col_avg_top5_worst": dict(sorted(col_avg.items(), key=lambda kv: -kv[1])[:5]),
        "reasons": reasons,
    }


# -------- Gate 2: Concat IC --------
def concat_ic(arrs: Dict[str, np.ndarray]) -> Dict:
    """Compute concat IC (Pearson, finite mask) across the four standard horizons."""
    pairs = [
        ("pred_log_ret_1s",  "target_log_ret_1s"),
        ("pred_log_ret_5s",  "target_log_ret_5s"),
        ("pred_log_ret_10s", "target_log_ret_10s"),
        ("pred_log_ret_30s", "target_log_ret_30s"),
    ]
    ics: Dict[str, float] = {}
    for p, t in pairs:
        if p not in arrs or t not in arrs:
            ics[p] = float("nan")
            continue
        x = arrs[p].astype(np.float64)
        y = arrs[t].astype(np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 100:
            ics[p] = float("nan")
            continue
        xs, ys = x[m], y[m]
        if xs.std() == 0 or ys.std() == 0:
            ics[p] = 0.0
            continue
        ics[p] = float(np.corrcoef(xs, ys)[0, 1])

    ic10 = ics.get("pred_log_ret_10s", float("nan"))
    vs_baseline = ic10 - BASELINE_CONCAT_IC_10S if not np.isnan(ic10) else float("nan")
    return {
        "concat_ic_per_horizon": ics,
        "concat_ic_10s": ic10,
        "vs_baseline_0_106": vs_baseline,
        "pass": (not np.isnan(ic10)) and (ic10 >= BASELINE_CONCAT_IC_10S * 0.85),
    }


# -------- Gate 3/4: Symmetric L/S balance + regime --------
def _import_symmetric_helpers():
    """Lazy import — these touch heavy deps + market replay."""
    from stream_continuation_backtest import NON_DIRECTIONAL_HEADS  # noqa
    from hc432_fifo_full_market_replay import run_one_date  # noqa
    return NON_DIRECTIONAL_HEADS, run_one_date


def directional_signal_v342(name: str, vals: np.ndarray, NDH: set) -> np.ndarray:
    if name in NDH:
        return np.zeros_like(vals)
    if name.startswith("pred_p_reversal_"):
        return np.zeros_like(vals)
    if name in ("pred_pred_mae_30s_ticks", "pred_pred_mae_60s_ticks"):
        return -vals
    return vals


def calibrate_equal_count_thresholds(signal: np.ndarray, is_mask: np.ndarray,
                                     per_side_rate: float = 0.05) -> Tuple[float, float]:
    is_signal = signal[is_mask & np.isfinite(signal)]
    if is_signal.size == 0:
        return 0.0, 0.0
    K = max(1, int(per_side_rate * is_signal.size))
    pos = is_signal[is_signal > 0]
    neg = is_signal[is_signal < 0]
    if pos.size >= K:
        k_pos = float(np.partition(pos, -K)[-K])
    elif pos.size > 0:
        k_pos = float(pos.min())
    else:
        k_pos = float("inf")
    if neg.size >= K:
        k_neg = float(-np.partition(neg, K - 1)[K - 1])
    elif neg.size > 0:
        k_neg = float(-neg.max())
    else:
        k_neg = float("inf")
    return k_pos, k_neg


def calibrate_nondirectional_threshold(signal: np.ndarray, is_mask: np.ndarray,
                                        target_pct: float = 0.05) -> float:
    is_signal = signal[is_mask & np.isfinite(signal)]
    if is_signal.size == 0:
        return 0.0
    return float(np.quantile(np.abs(is_signal), 1.0 - target_pct))


def symmetric_confluence_mask(arrs, heads, is_mask, NDH) -> Tuple[np.ndarray, np.ndarray, Dict]:
    n = arrs[heads[0]].shape[0]
    long_mask = np.ones(n, dtype=bool)
    short_mask = np.ones(n, dtype=bool)
    calib: Dict = {}
    for h in heads:
        if h not in arrs:
            raise KeyError(f"NPZ missing head {h}")
        raw = arrs[h].astype(np.float64)
        signal = directional_signal_v342(h, raw, NDH)
        if h in NDH:
            k = calibrate_nondirectional_threshold(signal, is_mask, target_pct=0.05)
            calib[h] = {"kind": "nondirectional", "k": k}
            keep = np.isfinite(signal) & (np.abs(signal) > k)
            long_mask &= keep
            short_mask &= keep
        else:
            k_pos, k_neg = calibrate_equal_count_thresholds(signal, is_mask, per_side_rate=0.05)
            calib[h] = {"kind": "directional", "k_pos": k_pos, "k_neg": k_neg}
            finite = np.isfinite(signal)
            long_mask &= finite & (signal > k_pos)
            short_mask &= finite & (signal < -k_neg)
    return long_mask, short_mask, calib


def run_one_config_replay(arrs, cfg_type, cfg_name, heads, is_mask, NDH, run_one_date):
    long_mask, short_mask, calib = symmetric_confluence_mask(arrs, heads, is_mask, NDH)
    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())
    log(f"[{cfg_name}] long_triggers={n_long} short_triggers={n_short} "
        f"short_share_triggers={n_short/max(1,n_long+n_short):.3f}")

    sample_dates = arrs["sample_dates"]
    unique_dates = sorted(set(sample_dates.tolist()))
    all_fills: List[dict] = []
    for date_str in unique_dates:
        day_mask = sample_dates == date_str
        for side, mask in [("long", long_mask), ("short", short_mask)]:
            sel = day_mask & mask
            if not sel.any():
                continue
            idx_in_day = np.flatnonzero(sel[day_mask])
            strength = np.ones(idx_in_day.size, dtype=np.float64)
            try:
                fills = run_one_date(
                    date_str=date_str, idx_in_day=idx_in_day, direction=side,
                    strength=strength, tp_ticks=TP_TICKS, sl_ticks=SL_TICKS,
                    hold_s=HOLD_S, cancel_s=CANCEL_S, order_type=ORDER_TYPE,
                )
            except Exception as e:
                log(f"WARN replay {date_str} {side}: {e}")
                continue
            for f in fills:
                if "error" in f:
                    continue
                f["config"] = cfg_name
                f["config_type"] = cfg_type
                f["side"] = side
                all_fills.append(f)

    fills_df = pd.DataFrame(all_fills)
    return fills_df, calib, n_long, n_short


def summarize_config(fills_df: pd.DataFrame, cfg_name: str) -> Dict:
    if fills_df.empty:
        return {"config": cfg_name, "n_fills": 0, "Sharpe": 0.0, "Sortino": 0.0,
                "PF": 0.0, "WR": 0.0, "long_share": 0.0, "short_share": 0.0,
                "day_conc": float("nan"), "Sharpe_green": float("nan"),
                "Sharpe_red": float("nan"), "regime_skew": float("nan"),
                "pass_dayconc": False, "pass_regime": False, "pass_ls_balance": False}
    net = fills_df["net_ticks"].values
    n = len(net)
    mean_t = float(net.mean())
    std_t = float(net.std(ddof=1)) if n > 1 else 0.0
    sharpe = (mean_t / std_t) if std_t > 0 else 0.0
    downside = net[net < 0]
    d_std = float(downside.std(ddof=1)) if downside.size > 1 else 0.0
    sortino = (mean_t / d_std) if d_std > 0 else 0.0
    gains = net[net > 0].sum()
    losses = -net[net < 0].sum()
    pf = (gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)
    wr = float((net > 0).mean())
    n_long = int(fills_df["side"].eq("long").sum())
    n_short = int(fills_df["side"].eq("short").sum())
    long_share = n_long / max(1, n)
    short_share = n_short / max(1, n)

    # Per-day for day_conc + regime
    day_groups = fills_df.groupby("date")["net_ticks"].agg(["sum", "count", "mean", "std"])
    day_net_total = day_groups["sum"]
    day_conc = float(day_net_total.max() / day_net_total.sum()) if day_net_total.sum() > 0 else float("nan")

    day_regime = {d: regime_classify(str(d)) for d in day_groups.index}
    day_sharpes = []
    for d in day_groups.index:
        sub = fills_df[fills_df["date"] == d]["net_ticks"].values
        if sub.size > 1 and sub.std(ddof=1) > 0:
            day_sharpes.append((d, float(sub.mean() / sub.std(ddof=1)), day_regime[d]))
    green_s = [s for _, s, r in day_sharpes if r == "green"]
    red_s = [s for _, s, r in day_sharpes if r == "red"]
    green_mean = float(np.mean(green_s)) if green_s else float("nan")
    red_mean = float(np.mean(red_s)) if red_s else float("nan")
    if not (np.isnan(green_mean) or np.isnan(red_mean)):
        mx = max(abs(green_mean), abs(red_mean))
        skew = abs(green_mean - red_mean) / mx if mx > 0 else 0.0
    else:
        skew = float("nan")

    return {
        "config": cfg_name,
        "n_fills": n,
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "PF": float(pf),
        "WR": float(wr),
        "long_share": float(long_share),
        "short_share": float(short_share),
        "day_conc": float(day_conc) if not np.isnan(day_conc) else float("nan"),
        "Sharpe_green": green_mean,
        "Sharpe_red": red_mean,
        "regime_skew": skew,
        "pass_dayconc": (not np.isnan(day_conc)) and (day_conc <= DAY_CONC_MAX),
        "pass_regime": (not np.isnan(skew)) and (abs(skew) <= REGIME_SKEW_MAX),
        "pass_ls_balance": (LONG_SHARE_LO <= long_share <= LONG_SHARE_HI),
    }


def run_symmetric_gate(arrs: Dict[str, np.ndarray]) -> Dict:
    NDH, run_one_date = _import_symmetric_helpers()
    sample_dates = arrs["sample_dates"]
    unique_dates = sorted(set(sample_dates.tolist()))
    n_is = max(1, int(IS_FRACTION * len(unique_dates)))
    is_dates = set(unique_dates[:n_is])
    is_mask = np.isin(sample_dates, list(is_dates))
    log(f"[sym-gate] {len(unique_dates)} dates total, {n_is} IS dates")

    summaries: List[Dict] = []
    all_fills: List[pd.DataFrame] = []
    for cfg_type, cfg_name, heads in CONFIGS:
        try:
            fills_df, calib, n_long, n_short = run_one_config_replay(
                arrs, cfg_type, cfg_name, heads, is_mask, NDH, run_one_date
            )
        except Exception as e:
            log(f"ERR cfg {cfg_name}: {e}")
            summaries.append({"config": cfg_name, "error": str(e),
                              "n_fills": 0, "Sharpe": 0.0, "Sortino": 0.0, "PF": 0.0, "WR": 0.0,
                              "long_share": 0.0, "short_share": 0.0,
                              "day_conc": float("nan"), "regime_skew": float("nan"),
                              "Sharpe_green": float("nan"), "Sharpe_red": float("nan"),
                              "pass_dayconc": False, "pass_regime": False, "pass_ls_balance": False})
            continue
        s = summarize_config(fills_df, cfg_name)
        s["n_long_triggers"] = n_long
        s["n_short_triggers"] = n_short
        summaries.append(s)
        if not fills_df.empty:
            all_fills.append(fills_df)

    df = pd.DataFrame(summaries)
    if not df.empty:
        df.to_parquet(OUT_DIR / "fold0_symmetric_gate_summary.parquet", index=False)
    if all_fills:
        pd.concat(all_fills, ignore_index=True).to_parquet(
            OUT_DIR / "fold0_symmetric_gate_fills.parquet", index=False)

    return {"per_config": summaries, "summary_df": df}


# -------- Gate 5: MFE-within-horizon --------
def mfe_within_horizon_check(arrs: Dict[str, np.ndarray]) -> Dict:
    """For each config, check TP ≤ p90(MFE within h), hold ≤ 1.5h, cancel ≤ h.
    Approximates realized MFE within horizon h from `pred_pred_mfe_*_ticks` if
    available (model's own MFE head ≠ realized MFE — flag AMBER). If realized
    MFE columns absent, mark AMBER with a TO-DO."""
    results: Dict[str, Dict] = {}
    any_amber = False
    any_fail = False
    for _, cfg_name, _ in CONFIGS:
        h_s = CONFIG_PRIMARY_HORIZON_S[cfg_name]
        # The NPZ contains target_pred_mfe_30s_ticks and target_pred_mfe_60s_ticks
        # which are the *realized* MFE values within 30s and 60s windows (the
        # `target_` prefix denotes the supervised label; `pred_mfe_*` is the head name).
        # Pick the tightest horizon ≥ h_s as upper-bound proxy. For h_s≤30 use 30s realized;
        # for h_s>30 use 60s realized. If neither exists fall back to AMBER.
        proxy_key = None
        if h_s <= 30 and "target_pred_mfe_30s_ticks" in arrs:
            proxy_key = "target_pred_mfe_30s_ticks"
        elif "target_pred_mfe_60s_ticks" in arrs:
            proxy_key = "target_pred_mfe_60s_ticks"
        elif "target_pred_mfe_30s_ticks" in arrs:
            proxy_key = "target_pred_mfe_30s_ticks"
        if proxy_key is None:
            results[cfg_name] = {
                "horizon_s": h_s,
                "tp_ticks": TP_TICKS,
                "hold_s": HOLD_S,
                "cancel_s": CANCEL_S,
                "tp_p90_mfe_h": None,
                "tp_within_p90": None,
                "hold_within_1_5h": HOLD_S <= 1.5 * h_s,
                "cancel_within_h": CANCEL_S <= h_s,
                "status": "AMBER",
                "todo": "realized MFE within horizon h not available in NPZ; "
                        "needs MBO replay MFE/MAE table",
            }
            any_amber = True
            # Still enforce hold/cancel rule-based checks
            if HOLD_S > 1.5 * h_s or CANCEL_S > h_s:
                any_fail = True
                results[cfg_name]["status"] = "FAIL"
            continue

        mfe = arrs[proxy_key].astype(np.float64)
        finite = np.isfinite(mfe)
        p90 = float(np.quantile(mfe[finite], 0.90)) if finite.any() else float("nan")
        tp_within = (not np.isnan(p90)) and TP_TICKS <= p90
        hold_within = HOLD_S <= 1.5 * h_s
        cancel_within = CANCEL_S <= h_s
        status = "PASS"
        if not (tp_within and hold_within and cancel_within):
            status = "FAIL"
            any_fail = True
        results[cfg_name] = {
            "horizon_s": h_s,
            "tp_ticks": TP_TICKS,
            "tp_p90_mfe_h": p90,
            "tp_within_p90": tp_within,
            "hold_s": HOLD_S,
            "hold_within_1_5h": hold_within,
            "cancel_s": CANCEL_S,
            "cancel_within_h": cancel_within,
            "proxy_key": proxy_key,
            "status": status,
        }

    overall = "AMBER" if any_amber and not any_fail else ("FAIL" if any_fail else "PASS")
    return {"per_config": results, "overall": overall}


# -------- Gate 6: Delta vs broken baseline --------
def delta_vs_broken_baseline(treatment_df: pd.DataFrame) -> Dict:
    if not BROKEN_BASELINE_SUMMARY.exists():
        return {"available": False, "n_positive": 0, "deltas": {}}
    base = pd.read_parquet(BROKEN_BASELINE_SUMMARY)
    deltas: Dict[str, float] = {}
    for _, _, _ in []:
        pass
    cfgs = sorted(set(treatment_df["config"].tolist()) & set(base["config"].tolist()))
    n_pos = 0
    for cfg in cfgs:
        t = treatment_df[treatment_df["config"] == cfg]["Sharpe"].iloc[0]
        b = base[base["config"] == cfg]["Sharpe"].iloc[0]
        d = float(t - b)
        deltas[cfg] = d
        if d > 0:
            n_pos += 1
    return {"available": True, "n_positive": n_pos, "n_compared": len(cfgs), "deltas": deltas}


# -------- Discord briefing (HC #433) --------
def write_discord_briefing(verdict: Dict) -> None:
    status = verdict["status"]
    best_cfg = verdict.get("best_config") or "none"
    best_sharpe = verdict.get("best_sharpe", 0.0)
    ic_10s = verdict.get("concat_ic_10s")
    ls_pass = verdict.get("ls_balance_pass", False)
    regime_pass = verdict.get("regime_pass", False)
    n_positive = verdict.get("delta_vs_broken_baseline_n_positive", 0)

    headline = {
        "PASS": "Model retrain passed all deploy gates.",
        "FAIL": "Model retrain failed the deploy gates.",
        "AMBER": "Model retrain is partial pass; one gate inconclusive.",
        "FAIL_NAN_AUDIT": "Model retrain failed the data-quality audit.",
    }.get(status, f"Model retrain verdict: {status}.")

    ic_line = f"Signal strength (10s): {ic_10s:+.3f}" if isinstance(ic_10s, (int, float)) and ic_10s == ic_10s else "Signal strength: n/a"
    ls_line = "Long/short balance: OK" if ls_pass else "Long/short balance: skewed"
    regime_line = "Regime balance: OK" if regime_pass else "Regime balance: skewed"
    baseline_line = f"Beats previous broken-label run on {n_positive} of 5 configs."

    if status == "PASS":
        next_step = f"Promoting {best_cfg} (Sharpe {best_sharpe:+.2f}) to next-phase validation."
    elif status == "AMBER":
        next_step = "Running MBO replay to fill the missing MFE check, then re-verdict."
    else:
        next_step = "Holding deploy. Investigating label/feature quality before next retrain."

    lines = [headline, ic_line, ls_line, regime_line, baseline_line,
             f"Best config: {best_cfg} (Sharpe {best_sharpe:+.2f}).",
             next_step]
    DISCORD_PATH.write_text("\n".join(lines) + "\n")
    log(f"Wrote discord briefing: {DISCORD_PATH}")


# -------- Main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Re-run even if verdict.json exists")
    ap.add_argument("--self-test", action="store_true", help="Smoke-test against baseline NPZ")
    ap.add_argument("--max-wait-min", type=int, default=120, help="Max wait minutes for NPZ")
    ap.add_argument("--sleep-s", type=int, default=60, help="Poll sleep seconds")
    ap.add_argument("--no-wait", action="store_true",
                    help="If NPZ missing, exit 0 immediately (for cron polling guard)")
    ap.add_argument("--no-replay", action="store_true",
                    help="Skip the FIFO market-replay portion (sym gate stub-only). "
                         "Used by self-test for fast schema validation.")
    args = ap.parse_args()

    # Idempotency — NPZ-mtime aware. Self-test verdicts (or stale verdicts
    # from older NPZs) must NOT block re-run when a NEWER real fold-0 NPZ arrives.
    if VERDICT_PATH.exists() and not args.force and not args.self_test:
        verdict_mtime = VERDICT_PATH.stat().st_mtime
        if FOLD0_NPZ.exists():
            npz_mtime = FOLD0_NPZ.stat().st_mtime
            if npz_mtime > verdict_mtime:
                log(f"FOLD-0 NPZ is newer than existing verdict — re-running (npz_mtime={npz_mtime} > verdict_mtime={verdict_mtime})")
                # Fall through to evaluation
            else:
                log(f"verdict.json up-to-date (NPZ mtime={npz_mtime} <= verdict mtime={verdict_mtime}) — skipping (use --force to re-run)")
                return 0
        else:
            # Real fold-0 NPZ doesn't exist yet but a verdict does (probably from --self-test).
            # In --no-wait mode this is the normal "nothing to do yet" state.
            if args.no_wait:
                return 0
            log(f"verdict.json exists at {VERDICT_PATH} but FOLD0_NPZ absent — skipping (use --force to re-run)")
            return 0

    # Pick NPZ
    if args.self_test:
        npz_path = SELF_TEST_NPZ
        if not npz_path.exists():
            log(f"SELF-TEST FAIL: baseline NPZ not at {npz_path}")
            return 2
        log(f"SELF-TEST MODE — using {npz_path}")
    else:
        npz_path = FOLD0_NPZ
        if args.no_wait:
            if not npz_path.exists():
                # Silent no-op for cron polling guard
                return 0
        else:
            ok = wait_for_npz(npz_path, max_wait_s=args.max_wait_min * 60, sleep_s=args.sleep_s)
            if not ok:
                log("ABORT: timeout waiting for fold-0 NPZ")
                return 3

    # Load
    t0 = time.time()
    log(f"Loading NPZ {npz_path}")
    arrs = {k: v for k, v in np.load(npz_path, allow_pickle=False).items()}
    log(f"Loaded in {time.time()-t0:.1f}s — n_samples={list(arrs.values())[0].shape[0]:,} "
        f"n_keys={len(arrs)}")

    verdict: Dict = {
        "status": None,
        "npz_path": str(npz_path),
        "self_test": bool(args.self_test),
        "timestamp": datetime.now().isoformat(),
    }

    # Gate 1: NaN audit
    log("Running NaN audit (HC #485 R3)...")
    nan_res = nan_audit(arrs)
    verdict["nan_audit"] = nan_res
    if nan_res["status"] == "FAIL_NAN_AUDIT":
        verdict["status"] = "FAIL_NAN_AUDIT"
        verdict["verdict_reason"] = (
            f"Fold-0 NPZ failed NaN audit: " + "; ".join(nan_res.get("reasons", []))
        )
        VERDICT_PATH.write_text(json.dumps(verdict, indent=2, default=str))
        write_discord_briefing(verdict)
        log("HALT — NaN audit failed")
        append_run_history(f"eval_fold0_v3_4_2_hc485: HALT NaN audit ({npz_path.name})")
        return 1

    # Gate 2: Concat IC
    log("Running concat IC gate...")
    ic_res = concat_ic(arrs)
    verdict["concat_ic"] = ic_res
    verdict["concat_ic_10s"] = ic_res["concat_ic_10s"]
    verdict["vs_baseline_0_106"] = ic_res["vs_baseline_0_106"]
    verdict["ic_pass"] = ic_res["pass"]
    log(f"  concat IC 10s = {ic_res['concat_ic_10s']:.4f} "
        f"(baseline 0.106, delta {ic_res['vs_baseline_0_106']:+.4f}) "
        f"pass={ic_res['pass']}")

    # Gate 3/4: Symmetric L/S balance + regime gate (FIFO replay)
    if args.no_replay:
        log("Skipping FIFO replay (--no-replay)")
        sym_res = {"per_config": [], "summary_df": pd.DataFrame()}
    else:
        log("Running symmetric L/S gate + FIFO replay across 5 configs...")
        try:
            sym_res = run_symmetric_gate(arrs)
        except Exception as e:
            log(f"ERR sym-gate: {e}\n{traceback.format_exc()}")
            sym_res = {"per_config": [], "summary_df": pd.DataFrame(), "error": str(e)}

    verdict["symmetric_gate"] = {"per_config": sym_res.get("per_config", [])}
    sdf = sym_res.get("summary_df", pd.DataFrame())

    if sdf is not None and not sdf.empty:
        # Best config = highest Sharpe among rows passing all sub-gates
        passers = sdf[sdf["pass_dayconc"] & sdf["pass_regime"] & sdf["pass_ls_balance"]]
        if not passers.empty:
            best = passers.sort_values("Sharpe", ascending=False).iloc[0]
        else:
            best = sdf.sort_values("Sharpe", ascending=False).iloc[0]
        verdict["best_config"] = str(best["config"])
        verdict["best_sharpe"] = float(best["Sharpe"])
        verdict["day_conc"] = float(best["day_conc"]) if pd.notna(best["day_conc"]) else float("nan")
        verdict["regime_skew"] = float(best["regime_skew"]) if pd.notna(best["regime_skew"]) else float("nan")
        verdict["ls_balance_pass"] = bool(best["pass_ls_balance"])
        verdict["regime_pass"] = bool(best["pass_dayconc"] and best["pass_regime"])
    else:
        verdict["best_config"] = None
        verdict["best_sharpe"] = 0.0
        verdict["day_conc"] = float("nan")
        verdict["regime_skew"] = float("nan")
        verdict["ls_balance_pass"] = False
        verdict["regime_pass"] = False

    # Gate 5: MFE-within-horizon
    log("Running MFE-within-horizon check (HC #432 R2)...")
    mfe_res = mfe_within_horizon_check(arrs)
    verdict["mfe_within_horizon"] = mfe_res
    verdict["mfe_horizon_pass"] = (mfe_res["overall"] == "PASS")

    # Gate 6: Delta vs broken baseline
    log("Computing delta vs broken-label baseline...")
    if sdf is not None and not sdf.empty:
        delta_res = delta_vs_broken_baseline(sdf)
    else:
        delta_res = {"available": False, "n_positive": 0, "deltas": {}}
    verdict["delta_vs_broken_baseline"] = delta_res
    verdict["delta_vs_broken_baseline_n_positive"] = delta_res.get("n_positive", 0)

    # Final verdict
    gate_pass = [verdict["ic_pass"], verdict["ls_balance_pass"],
                 verdict["regime_pass"]]
    if mfe_res["overall"] == "AMBER":
        # AMBER permitted if other gates pass
        final = "AMBER" if all(gate_pass) else "FAIL"
    elif mfe_res["overall"] == "FAIL":
        final = "FAIL"
    else:
        final = "PASS" if all(gate_pass) else "FAIL"

    verdict["status"] = final
    reasons = []
    if not verdict["ic_pass"]:
        reasons.append(f"IC_10s {verdict['concat_ic_10s']:.3f} below 85% of baseline 0.106")
    if not verdict["ls_balance_pass"]:
        reasons.append("long/short share outside [0.20,0.80]")
    if not verdict["regime_pass"]:
        reasons.append(f"regime/day_conc fail (skew {verdict.get('regime_skew')}, "
                       f"day_conc {verdict.get('day_conc')})")
    if mfe_res["overall"] != "PASS":
        reasons.append(f"MFE-within-horizon: {mfe_res['overall']}")
    if final == "PASS":
        verdict["verdict_reason"] = (
            f"All deploy gates cleared on the fixed-label retrain; "
            f"best config {verdict['best_config']} at Sharpe {verdict['best_sharpe']:+.2f}.")
    elif final == "AMBER":
        verdict["verdict_reason"] = (
            "Gates cleared except MFE-within-horizon (needs MBO replay); "
            "candidate is provisional.")
    else:
        verdict["verdict_reason"] = "Deploy blocked: " + "; ".join(reasons)

    VERDICT_PATH.write_text(json.dumps(verdict, indent=2, default=str))
    log(f"Wrote {VERDICT_PATH}")

    write_discord_briefing(verdict)

    # Per HC #485 R5
    REGEN_COMPLETE.write_text(json.dumps({
        "completed_at": datetime.now().isoformat(),
        "verdict": final,
        "self_test": bool(args.self_test),
        "npz": str(npz_path),
    }, indent=2))
    log(f"Wrote {REGEN_COMPLETE}")

    append_run_history(
        f"eval_fold0_v3_4_2_hc485 {'(self-test)' if args.self_test else ''}: "
        f"{final} — best={verdict.get('best_config')} "
        f"Sharpe={verdict.get('best_sharpe'):+.2f}"
    )
    log(f"DONE — verdict={final} wall={time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
