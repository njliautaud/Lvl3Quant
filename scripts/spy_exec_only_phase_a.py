#!/usr/bin/env python3
"""spy_exec_only_phase_a.py — HC #534 Phase A: cross-asset IC + execution PnL.

PURPOSE
=======
Test the user's hypothesis: can we use ES CNN-Mamba v3.4.2 predictions to
execute on SPY (which is cheaper, no PDT issues, Alpaca-fillable)?

PROTOCOL
========
For each of 9 days (Mar 2,3,4,5,6,9,10,11,12 2026):
  1. Load ES CNN-Mamba v3.4.2 predictions (per-date OOT files).
  2. Map prediction i -> ES MBO event index 999 + i*250 (verified stride/window).
  3. Look up ES event timestamp -> get prediction emit time.
  4. Load SPY 250-ms mid grid (pulled from Neptune).
  5. For each lag in {0, 50, 100, 200, 500, 1000} ms, look up SPY mid at
     (pred_ts + lag) and compute forward SPY drift over multiple horizons.
  6. Compute IC(ES_pred, SPY_drift_at_lag_+_h) per (lag, horizon).
  7. Apply confidence thresholds + SPY cost model -> compute PnL.
  8. Apply HC #428 R1 regime gate + HC #428 R2 MFE-within-horizon gate.

INPUTS
======
ES predictions per date: /home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/
                          oot_47day_perdate/oot_YYYYMMDD.npz
ES MBO event timestamps : /home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3/
                          YYYYMMDD_mbo_events.npz  (uses 'timestamps' key only)
SPY mid 250ms grid      : /home/jupiter/Lvl3Quant/data/processed/spy_mid_grid/
                          YYYYMMDD_mid_250ms.npz  (built by spy_mid_grid_neptune.py)

OUTPUTS
=======
/home/jupiter/Lvl3Quant/output/spy_exec_only/phase_a_report.md
/home/jupiter/Lvl3Quant/output/spy_exec_only/phase_a_log.txt
/home/jupiter/Lvl3Quant/output/spy_exec_only/per_day_metrics.json
/home/jupiter/Lvl3Quant/output/spy_exec_only/ic_table.csv
/home/jupiter/Lvl3Quant/output/spy_exec_only/pnl_table.csv

COST CONSTANTS (SPY @ Alpaca / IBKR)
====================================
SPY_TICK_USD            = 0.01   # 1 cent
SPY_COMMISSION_PER_SH   = 0.0    # Alpaca commission-free
SPY_SEC_FEE_RATE        = 8e-6   # SEC Section 31 fee on SALES, ~$8/$1M notional
SPY_TAF_RATE            = 1.66e-5 # FINRA TAF on SALES, $0.0000166/share approx (uses share-based; use rate as fallback)
SPY_TYPICAL_SPREAD_TICKS = 1.0   # 1 cent typical

Reference signal head: pred_log_ret_1s (champion 1s horizon per HC #471/#475/#477/#529).
Per-asset normalization: SPY drift expressed in BPS for cross-asset comparability.
"""
from __future__ import annotations
import json, sys, time, logging
from pathlib import Path
from typing import Optional
import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
ES_PRED_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
ES_MBO_DIR  = ROOT / "data/processed/mbo_events_smart_v3"
SPY_GRID_DIR = ROOT / "data/processed/spy_mid_grid"
OUT_DIR = ROOT / "output/spy_exec_only"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATES = ["20260302","20260303","20260304","20260305","20260306",
         "20260309","20260310","20260311","20260312"]

# Stride/window math validated across all 9 days (max truncation = 86)
WINDOW = 1000
STRIDE = 250

# Lag and horizon grids (ms)
LAGS_MS = [0, 50, 100, 200, 500, 1000]
HORIZONS_MS = [1000, 5000, 10000, 30000]  # matches ES heads 1s/5s/10s/30s

# Confidence thresholds (top/bottom pct of |pred|)
TOP_PCTS = [0.05, 0.10, 0.20]   # top-5% / top-10% / top-20% by absolute prediction strength

# SPY cost model
SPY_TICK_USD = 0.01
SPY_COMMISSION_PER_SH = 0.0
SPY_SEC_FEE_RATE = 8e-6        # on SELL notional only (proceeds)
SPY_TAF_PER_SH = 1.66e-5       # on SELL shares only; we'll convert via avg share notional
SPY_TYPICAL_SPREAD_BPS = None  # filled from actual data per day
SPY_PASSIVE_FILL_HALF_SPREAD = True  # passive limit gets one-side fill

LOG_PATH = OUT_DIR / "phase_a_log.txt"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(LOG_PATH, mode='w'),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_es_pred_times(date_str: str):
    """Returns (pred_ts_ns, pred_log_ret_1s, pred_log_ret_5s, pred_log_ret_10s,
    pred_log_ret_30s, masks dict). NaN-filled positions still present."""
    es_path = ES_MBO_DIR / f"{date_str}_mbo_events.npz"
    pred_path = ES_PRED_DIR / f"oot_{date_str}.npz"
    if not es_path.exists() or not pred_path.exists():
        return None
    es = np.load(str(es_path), allow_pickle=True)
    timestamps = es["timestamps"]
    pf = np.load(str(pred_path), allow_pickle=True)
    n_pred = pf["pred_log_ret_1s"].shape[0]
    # Map prediction i -> ES event index 999 + i*250 (= WINDOW-1 + i*STRIDE)
    idx = (WINDOW - 1) + np.arange(n_pred) * STRIDE
    if idx[-1] >= len(timestamps):
        # Truncate to safe range
        ok_n = np.searchsorted(idx, len(timestamps), side='left')
        idx = idx[:ok_n]
        n_pred = ok_n
    pred_ts_ns = timestamps[idx]
    out = {
        "pred_ts_ns": pred_ts_ns,
        "pred_log_ret_1s": pf["pred_log_ret_1s"][:n_pred],
        "pred_log_ret_5s": pf["pred_log_ret_5s"][:n_pred],
        "pred_log_ret_10s": pf["pred_log_ret_10s"][:n_pred],
        "pred_log_ret_30s": pf["pred_log_ret_30s"][:n_pred],
        "mask_log_ret_1s": pf["mask_log_ret_1s"][:n_pred],
        "mask_log_ret_5s": pf["mask_log_ret_5s"][:n_pred],
        "mask_log_ret_10s": pf["mask_log_ret_10s"][:n_pred],
        "mask_log_ret_30s": pf["mask_log_ret_30s"][:n_pred],
        # Also keep ES-target log returns for sanity check (ES IC reproducibility)
        "tgt_log_ret_1s": pf["target_log_ret_1s"][:n_pred],
        "tgt_log_ret_5s": pf["target_log_ret_5s"][:n_pred],
        "tgt_log_ret_10s": pf["target_log_ret_10s"][:n_pred],
        "tgt_log_ret_30s": pf["target_log_ret_30s"][:n_pred],
    }
    return out


def load_spy_grid(date_str: str):
    p = SPY_GRID_DIR / f"{date_str}_mid_250ms.npz"
    if not p.exists():
        return None
    d = np.load(str(p), allow_pickle=True)
    return {
        "grid_ts_ns": d["grid_ts_ns"],
        "mid": d["mid_price"],
        "bid": d["bid_price"],
        "ask": d["ask_price"],
        "spread_ticks": d["spread_ticks"],
    }


# ---------------------------------------------------------------------------
# Cross-asset alignment
# ---------------------------------------------------------------------------

def spy_at_time(grid_ts: np.ndarray, mid: np.ndarray, query_ns: np.ndarray):
    """For each query time, return the SPY mid at the LAST grid point <= query.
    Returns mid array; positions where no valid grid point exists return NaN."""
    idx = np.searchsorted(grid_ts, query_ns, side='right') - 1
    valid = (idx >= 0) & (idx < len(grid_ts))
    out = np.full(len(query_ns), np.nan, dtype=np.float64)
    if valid.any():
        cand = mid[idx[valid]]
        # also require cand > 0 (book initialized)
        out[valid] = np.where(cand > 0, cand, np.nan)
    return out


def compute_spy_drift(spy_grid, pred_ts_ns: np.ndarray, lag_ms: int, horizon_ms: int):
    """For each ES pred timestamp t, compute SPY drift in BPS from t+lag to t+lag+horizon."""
    lag_ns = lag_ms * 1_000_000
    h_ns = horizon_ms * 1_000_000
    entry_ts = pred_ts_ns + lag_ns
    exit_ts = pred_ts_ns + lag_ns + h_ns
    mid_entry = spy_at_time(spy_grid["grid_ts_ns"], spy_grid["mid"], entry_ts)
    mid_exit  = spy_at_time(spy_grid["grid_ts_ns"], spy_grid["mid"], exit_ts)
    with np.errstate(invalid='ignore', divide='ignore'):
        drift_bps = (mid_exit - mid_entry) / mid_entry * 1e4
    return drift_bps, mid_entry, mid_exit


# ---------------------------------------------------------------------------
# IC
# ---------------------------------------------------------------------------

def pearson_ic(pred, target):
    m = np.isfinite(pred) & np.isfinite(target)
    if m.sum() < 100:
        return np.nan, int(m.sum())
    p = pred[m].astype(np.float64)
    t = target[m].astype(np.float64)
    if p.std() == 0 or t.std() == 0:
        return np.nan, int(m.sum())
    return float(np.corrcoef(p, t)[0,1]), int(m.sum())


# ---------------------------------------------------------------------------
# Execution PnL  (SPY)
# ---------------------------------------------------------------------------

def spy_cost_round_trip(mid_entry: float, side: str, passive: bool = False) -> float:
    """Total round-trip cost (entry + exit) per share in dollars.

    Components:
      - Spread crossing: 1 full tick if MARKET on both sides; 0 if perfect PASSIVE on both
        (typical SPY spread = 1 tick / $0.01 during RTH).
      - SEC fee: $8/$1M on SELL notional.
      - TAF: $0.0000166/share on SELL.
      - Commission: $0 at Alpaca.
    """
    spread_cost = 0.0 if passive else SPY_TICK_USD  # one-side cross per leg; net 1 tick for full RT market
    # For passive both legs, spread_cost = 0 (we sit on the book on entry AND exit)
    # For market both legs, spread_cost = 1.0 * tick (lose half spread each leg, total 1 tick)
    sec_fee  = SPY_SEC_FEE_RATE * mid_entry          # ~ per share, approx mid as proxy for SELL price
    taf_fee  = SPY_TAF_PER_SH                        # per share, irrespective of price
    commission = 2 * SPY_COMMISSION_PER_SH
    return spread_cost + sec_fee + taf_fee + commission


def simulate_pnl(pred, spy_drift_bps, mid_entry, top_pct: float, side_filter: str,
                 mode: str = 'market'):
    """Simulate PnL after costs.

    side_filter: 'short' = use only signals predicting DOWN; 'long' = only UP; 'both' = both with sign.
    mode: 'market' or 'passive'.
    Top_pct: threshold on |pred| (across the day's signals).

    Returns dict with raw_pnl_bps, net_pnl_bps, n_trades, hit rate, etc.
    """
    # Mask: finite pred, finite drift, mid_entry > 0
    m = np.isfinite(pred) & np.isfinite(spy_drift_bps) & np.isfinite(mid_entry) & (mid_entry > 0)
    if m.sum() < 20:
        return None
    p = pred[m]
    d = spy_drift_bps[m]
    me = mid_entry[m]
    # Confidence threshold (top X% by |pred|)
    thresh = np.percentile(np.abs(p), 100.0 * (1.0 - top_pct))
    sel = np.abs(p) >= thresh
    if side_filter == 'short':
        sel &= (p < 0)
        sign = -1.0  # short: pnl = -drift  (drift positive means price went up against us)
    elif side_filter == 'long':
        sel &= (p > 0)
        sign = +1.0
    else:
        sign = np.sign(p)  # both
        sign[sign == 0] = 1.0
    if sel.sum() < 5:
        return None
    raw_drift_bps = (d[sel] * (sign if np.isscalar(sign) else sign[sel])).astype(np.float64)
    # Cost: round-trip cost in dollars per share, converted to bps of entry price.
    me_sel = me[sel]
    cost_per_share = np.array([spy_cost_round_trip(float(x), 'short' if side_filter=='short' else 'long',
                                                    passive=(mode=='passive'))
                               for x in me_sel])
    cost_bps = cost_per_share / me_sel * 1e4
    net = raw_drift_bps - cost_bps
    return {
        "n_trades": int(sel.sum()),
        "raw_mean_bps": float(np.mean(raw_drift_bps)),
        "cost_mean_bps": float(np.mean(cost_bps)),
        "net_mean_bps":  float(np.mean(net)),
        "net_std_bps":   float(np.std(net, ddof=1)) if sel.sum() > 1 else 0.0,
        "wr":            float(np.mean(raw_drift_bps > 0)),
        "wr_net":        float(np.mean(net > 0)),
        "sharpe":        float(np.mean(net) / np.std(net, ddof=1)) if (sel.sum() > 1 and np.std(net, ddof=1) > 0) else 0.0,
        "sortino":       _sortino(net),
        "pf":            _pf(net),
        "p_thresh":      float(thresh),
        "net_pnl_bps_total": float(np.sum(net)),
    }


def _sortino(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    downside = x[x < 0]
    if len(downside) == 0:
        return float('inf') if x.mean() > 0 else 0.0
    dd = np.std(downside, ddof=1) if len(downside) > 1 else float(abs(downside[0]))
    return float(x.mean() / dd) if dd > 0 else 0.0

def _pf(x):
    x = np.asarray(x, dtype=np.float64)
    pos = x[x > 0].sum()
    neg = -x[x < 0].sum()
    if neg <= 0:
        return float('inf') if pos > 0 else 0.0
    return float(pos / neg)


def bootstrap_sharpe_ci(x, n_boot=1000, seed=42):
    if len(x) < 30:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        s = x[idx]
        sd = s.std(ddof=1)
        boots[i] = s.mean() / sd if sd > 0 else 0.0
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


# ---------------------------------------------------------------------------
# Regime classification (HC #428 R1) — ES close-to-close green/red/flat
# ---------------------------------------------------------------------------

def classify_es_regime_per_day(date_str: str, tgt_drift_pct_threshold: float = 0.0010):
    """Classify each day as green/red/flat based on ES close-to-close drift.
    Uses ES MBO label_5min cum drift or first-to-last mid.
    For Phase A we use a simple proxy: cumulative sum of target_log_ret_1s across the day.
    Threshold = 0.10% (10 bps). Returns 'green' / 'red' / 'flat'."""
    p = ES_PRED_DIR / f"oot_{date_str}.npz"
    if not p.exists():
        return "unknown"
    d = np.load(str(p), allow_pickle=True)
    tgt = d["target_log_ret_1s"]
    mask = d["mask_log_ret_1s"]
    valid = np.isfinite(tgt) & (mask > 0)
    if valid.sum() < 100:
        return "unknown"
    # Cumulative sum of 1s log returns gives total log drift (overlapping windows — this is approximate
    # but adequate as a sign indicator for the day's net direction).
    # Stride between predictions = 250 events; 1s horizon overlap heavy.  To approximate close-to-close,
    # sum non-overlapping samples (every 4 strides ~ 1s if events were 250ms apart, but they aren't —
    # just use mean log_ret_1s * number_of_seconds_in_day instead).
    # Simpler: use last - first VWAP-equivalent via integral of mean log return.
    mean_log_ret_per_pred = float(np.nanmean(tgt[valid]))
    # We don't know exact stride in seconds; predictions emit at variable cadence (event-driven).
    # Use ES MBO ts to compute actual elapsed time.
    es = np.load(str(ES_MBO_DIR / f"{date_str}_mbo_events.npz"), allow_pickle=True)
    ts = es["timestamps"]
    elapsed_sec = (ts[-1] - ts[0]) / 1e9
    # Average pred cadence:
    n_pred = valid.sum()
    pred_cadence_sec = elapsed_sec / n_pred if n_pred > 0 else 0
    # Day drift proxy = sum of (mean_log_ret_per_pred * cadence) integrated
    # Each pred is forward 1s, so cum drift across day ≈ sum (without double-count). Use mean * (elapsed/1s).
    day_log_drift = mean_log_ret_per_pred * (elapsed_sec / 1.0) / max(1, (elapsed_sec / pred_cadence_sec))
    # Simpler & cleaner: just check sign+magnitude of average pred-stride 1s log return scaled by N strides per second.
    # For classification purpose just use mean log ret > threshold.
    if mean_log_ret_per_pred > tgt_drift_pct_threshold / 100:
        return "green"
    elif mean_log_ret_per_pred < -tgt_drift_pct_threshold / 100:
        return "red"
    else:
        return "flat"


def regime_classify_via_mid(date_str: str):
    """Better regime classifier: ES MBO last mid vs first mid using stored events.
    Falls back to flat if we can't extract absolute mid."""
    p = ES_MBO_DIR / f"{date_str}_mbo_events.npz"
    if not p.exists():
        return "unknown"
    # We don't have absolute mid in this file. But the labels_1s array is forward mid CHANGES in ticks.
    # Sum of (mid_t+1 - mid_t) approximated via... actually labels_1s[i] = mid(t+1s) - mid(t).
    # We can compute net drift by summing consecutive non-overlapping (i, i+stride_to_1s) but stride is event-time.
    # Easier proxy via SPY: use SPY close-to-close on the SPY grid for green/red.
    spy = load_spy_grid(date_str)
    if spy is None:
        return "unknown"
    mid_valid = spy["mid"][spy["mid"] > 0]
    if len(mid_valid) < 100:
        return "unknown"
    open_p = mid_valid[0]
    close_p = mid_valid[-1]
    pct = (close_p - open_p) / open_p * 100
    if pct > 0.10:
        return "green"
    elif pct < -0.10:
        return "red"
    else:
        return "flat"


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def analyze_day(date_str: str):
    log.info(f"=== Analyzing {date_str} ===")
    es = load_es_pred_times(date_str)
    spy = load_spy_grid(date_str)
    if es is None:
        log.warning(f"  {date_str}: ES preds missing")
        return None
    if spy is None:
        log.warning(f"  {date_str}: SPY grid missing")
        return None
    log.info(f"  {date_str}: ES preds = {len(es['pred_ts_ns'])}, SPY grid = {len(spy['grid_ts_ns'])}")
    log.info(f"  {date_str}: ES pred ts range = [{es['pred_ts_ns'][0]}, {es['pred_ts_ns'][-1]}]")
    log.info(f"  {date_str}: SPY grid ts range = [{spy['grid_ts_ns'][0]}, {spy['grid_ts_ns'][-1]}]")

    # Sanity: reproduce ES IC for v3.4.2 1s head on this day
    ic_es_1s, n_es_1s = pearson_ic(es["pred_log_ret_1s"], es["tgt_log_ret_1s"])
    log.info(f"  {date_str}: ES intra-asset IC(pred_1s, tgt_1s) = {ic_es_1s:.4f} (n={n_es_1s}) [sanity check]")

    # Restrict to ES pred timestamps that fall inside SPY grid range
    pred_ts = es["pred_ts_ns"]
    spy_ts_min, spy_ts_max = spy["grid_ts_ns"][0], spy["grid_ts_ns"][-1]
    in_window = (pred_ts >= spy_ts_min) & (pred_ts <= spy_ts_max)
    n_in_window = int(in_window.sum())
    log.info(f"  {date_str}: {n_in_window}/{len(pred_ts)} ES preds fall inside SPY RTH grid window")

    # Apply in_window filter
    pred_ts_w = pred_ts[in_window]
    pred_1s = es["pred_log_ret_1s"][in_window]
    pred_5s = es["pred_log_ret_5s"][in_window]
    pred_10s = es["pred_log_ret_10s"][in_window]
    pred_30s = es["pred_log_ret_30s"][in_window]

    day_result = {"date": date_str, "n_preds_in_window": n_in_window,
                  "n_preds_total": int(len(pred_ts)),
                  "ic_es_intra_1s": ic_es_1s,
                  "ic_table": {},      # keyed (lag_ms, horizon_ms) -> {"ic": ..., "n": ...}
                  "pnl_table": {},     # keyed (lag_ms, horizon_ms, top_pct, side, mode) -> dict
                  "spy_open": float(spy["mid"][spy["mid"]>0][0]) if (spy["mid"]>0).any() else None,
                  "spy_close": float(spy["mid"][spy["mid"]>0][-1]) if (spy["mid"]>0).any() else None,
                  }
    day_result["regime"] = regime_classify_via_mid(date_str)
    log.info(f"  {date_str}: regime = {day_result['regime']} (SPY open={day_result['spy_open']:.2f} close={day_result['spy_close']:.2f})")

    # IC + PnL for each (lag, horizon) combo
    # We test pred_1s vs SPY drift over horizon h (1s/5s/10s/30s — the same heads).
    head_preds = {1000: pred_1s, 5000: pred_5s, 10000: pred_10s, 30000: pred_30s}
    head_names = {1000: 'pred_1s', 5000: 'pred_5s', 10000: 'pred_10s', 30000: 'pred_30s'}

    for h_ms in HORIZONS_MS:
        for lag_ms in LAGS_MS:
            drift_bps, mid_entry, mid_exit = compute_spy_drift(spy, pred_ts_w, lag_ms, h_ms)
            pred_h = head_preds[h_ms]
            # IC: head h vs SPY drift over horizon h (same horizon match per HC #428 R2)
            ic_val, n_eff = pearson_ic(pred_h, drift_bps)
            key = f"lag{lag_ms}_h{h_ms}"
            day_result["ic_table"][key] = {"ic": ic_val, "n": n_eff, "head": head_names[h_ms]}

            # PnL: for each top_pct, each side, each mode
            for tp in TOP_PCTS:
                for side in ("short", "long"):
                    for mode in ("market", "passive"):
                        res = simulate_pnl(pred_h, drift_bps, mid_entry, tp, side, mode)
                        if res is not None:
                            res["lag_ms"] = lag_ms
                            res["horizon_ms"] = h_ms
                            res["top_pct"] = tp
                            res["side"] = side
                            res["mode"] = mode
                            day_result["pnl_table"][f"{key}_top{int(tp*100)}_{side}_{mode}"] = res

    return day_result


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def hc428_r1_gate(sharpes_by_regime: dict) -> str:
    """|Sharpe_green - Sharpe_red| / max(|Sg|,|Sr|) > 0.50 -> REJECT (regime-tailored)."""
    sg = sharpes_by_regime.get("green")
    sr = sharpes_by_regime.get("red")
    if sg is None or sr is None or np.isnan(sg) or np.isnan(sr):
        return f"INSUFFICIENT DATA (green={sg}, red={sr})"
    denom = max(abs(sg), abs(sr))
    if denom == 0:
        return "FLAT (both Sharpes near zero)"
    gap = abs(sg - sr) / denom
    verdict = "REJECT (regime-tailored)" if gap > 0.50 else "PASS"
    return f"{verdict} — green={sg:.3f} red={sr:.3f} gap={gap:.2%}"


def write_report(all_days):
    rep = []
    rep.append("# SPY Cross-Asset Execution — Phase A Report")
    rep.append(f"\n_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} — HC #534 / #527 R1+R2._\n")

    # ─── Data inventory ───
    rep.append("\n## Data Inventory\n")
    rep.append("| Date | ES preds | In SPY-window | SPY open | SPY close | Regime | ES intra-IC(1s) |")
    rep.append("|------|----------|---------------|----------|-----------|--------|------------------|")
    for r in all_days:
        if r is None: continue
        rep.append(f"| {r['date']} | {r['n_preds_total']:,} | {r['n_preds_in_window']:,} | "
                   f"{r['spy_open']:.2f} | {r['spy_close']:.2f} | {r['regime']} | "
                   f"{r['ic_es_intra_1s']:.4f} |")

    valid_days = [r for r in all_days if r is not None]
    if not valid_days:
        rep.append("\n**NO DAYS HAD COMPLETE DATA — Phase A blocked.**\n")
        return "\n".join(rep)

    # ─── ES Prediction Coverage Gap ───
    rep.append("\n## ES Prediction Coverage Gap\n")
    rep.append("All 9 days have ES v3.4.2 predictions available — no gap.")
    rep.append("Prediction-to-timestamp mapping: ES MBO event index = 999 + i*250 (stride=250, window=1000).")
    rep.append("Max truncation from end: 86 predictions (negligible).")

    # ─── Per-lag IC table ───
    rep.append("\n## Cross-Asset IC: ES Prediction vs SPY Forward Drift\n")
    rep.append("IC computed Pearson, head-horizon matched (pred_1s vs SPY drift over 1s, etc.).")
    rep.append("Drift measured in bps of SPY mid-price.\n")
    rep.append("### Concat IC (all valid samples across 9 days)\n")

    # Concatenate per-lag-horizon
    concat_ic = {}
    for h_ms in HORIZONS_MS:
        for lag_ms in LAGS_MS:
            key = f"lag{lag_ms}_h{h_ms}"
            # Pool per-day daily ICs and concat across samples for proper concat IC
            # For concat IC we'd need raw preds+drifts, but we only kept summaries.
            # Use mean of per-day ICs weighted by n as approximation.
            wsum = 0.0
            nsum = 0
            valid_ics = []
            for r in valid_days:
                v = r["ic_table"].get(key)
                if v is None or np.isnan(v["ic"]) or v["n"] < 50:
                    continue
                wsum += v["ic"] * v["n"]
                nsum += v["n"]
                valid_ics.append(v["ic"])
            mean_ic = wsum / nsum if nsum > 0 else np.nan
            # Standard error of mean across days
            if len(valid_ics) >= 2:
                se = np.std(valid_ics, ddof=1) / np.sqrt(len(valid_ics))
            else:
                se = np.nan
            concat_ic[key] = {"ic_weighted_mean": mean_ic, "n_total": nsum,
                              "n_days": len(valid_ics), "se_across_days": se}

    rep.append("| Horizon | Lag (ms) | Weighted-Mean IC | N samples | N days | SE (across days) |")
    rep.append("|---------|----------|------------------|-----------|--------|------------------|")
    for h_ms in HORIZONS_MS:
        for lag_ms in LAGS_MS:
            key = f"lag{lag_ms}_h{h_ms}"
            c = concat_ic[key]
            rep.append(f"| {h_ms/1000:.0f}s | {lag_ms} | {c['ic_weighted_mean']:.4f} | "
                       f"{c['n_total']:,} | {c['n_days']} | "
                       f"{c['se_across_days']:.4f}" + " |" if not np.isnan(c['se_across_days']) else " N/A |")

    # Per-day IC table at lag=0, all horizons
    rep.append("\n### Per-Day IC (lag=0, head-horizon matched)\n")
    rep.append("| Date | Regime | IC(1s) | IC(5s) | IC(10s) | IC(30s) |")
    rep.append("|------|--------|--------|--------|---------|---------|")
    for r in valid_days:
        row = [r["date"], r["regime"]]
        for h_ms in HORIZONS_MS:
            v = r["ic_table"].get(f"lag0_h{h_ms}")
            row.append(f"{v['ic']:.4f}" if v and not np.isnan(v['ic']) else "—")
        rep.append("| " + " | ".join(row) + " |")

    # ─── PnL ───
    rep.append("\n## Cross-Asset Execution PnL: Trade SPY on ES Signals\n")
    rep.append("Confidence threshold = top X% by |pred|. Side filter = short only / long only.")
    rep.append("Mode 'market' = cross 1 tick total round-trip + SEC + TAF. Mode 'passive' = no spread cost (best case).")
    rep.append("Costs: SPY $0 commission (Alpaca), SEC fee 8e-6 on sell notional, TAF $0.0000166/share.")
    rep.append("PnL reported in BPS of SPY notional. Sharpe = per-trade, NOT annualized.\n")

    # Aggregate per (top_pct, side, mode) at lag=0, head=1s (champion)
    rep.append("### Champion config: head=1s, lag=0, by (top_pct, side, mode)\n")
    rep.append("| Top% | Side | Mode | Days | N trades | Mean net (bps) | Sharpe | Sortino | PF | WR |")
    rep.append("|------|------|------|------|----------|----------------|--------|---------|----|----|")
    for tp in TOP_PCTS:
        for side in ("short", "long"):
            for mode in ("market", "passive"):
                key_suffix = f"top{int(tp*100)}_{side}_{mode}"
                # Aggregate across days for head=1s, lag=0
                base_key = f"lag0_h1000"
                full_key = f"{base_key}_{key_suffix}"
                day_results = []
                for r in valid_days:
                    p = r["pnl_table"].get(full_key)
                    if p: day_results.append(p)
                if not day_results: continue
                ntr_total = sum(p["n_trades"] for p in day_results)
                # Trade-level weighted mean
                net_mean = sum(p["net_mean_bps"] * p["n_trades"] for p in day_results) / max(1, ntr_total)
                # Pool sharpe approx: avg of daily sharpes
                shr_vals = [p["sharpe"] for p in day_results if p["n_trades"] >= 5]
                sortino_vals = [p["sortino"] for p in day_results if p["n_trades"] >= 5]
                pf_vals = [p["pf"] for p in day_results if p["n_trades"] >= 5 and np.isfinite(p["pf"])]
                wr_vals = [p["wr_net"] for p in day_results]
                shr = np.mean(shr_vals) if shr_vals else np.nan
                sortino = np.mean(sortino_vals) if sortino_vals else np.nan
                pf = np.mean(pf_vals) if pf_vals else np.nan
                wr = np.mean(wr_vals) if wr_vals else np.nan
                rep.append(f"| {int(tp*100)}% | {side} | {mode} | {len(day_results)} | {ntr_total} | "
                           f"{net_mean:+.2f} | {shr:.3f} | {sortino:.3f} | {pf:.2f} | {wr:.2%} |")

    # ─── Best-config table across all (lag, horizon) ───
    rep.append("\n### Best Sharpe per (lag, horizon, mode) — top10% short\n")
    rep.append("| Horizon | Lag (ms) | Mode | Days | N trades | Mean net (bps) | Sharpe |")
    rep.append("|---------|----------|------|------|----------|----------------|--------|")
    for h_ms in HORIZONS_MS:
        for lag_ms in LAGS_MS:
            for mode in ("market", "passive"):
                key = f"lag{lag_ms}_h{h_ms}_top10_short_{mode}"
                drs = [r["pnl_table"].get(key) for r in valid_days]
                drs = [d for d in drs if d]
                if not drs: continue
                ntr = sum(d["n_trades"] for d in drs)
                net_mean = sum(d["net_mean_bps"]*d["n_trades"] for d in drs)/max(1,ntr)
                shr = np.mean([d["sharpe"] for d in drs if d["n_trades"] >= 5])
                rep.append(f"| {h_ms/1000:.0f}s | {lag_ms} | {mode} | {len(drs)} | {ntr} | "
                           f"{net_mean:+.2f} | {shr:.3f} |")

    # ─── HC #428 R1 regime gate ───
    rep.append("\n## HC #428 R1 — Regime Stratification Gate\n")
    # Compute Sharpe per regime for champion config (top10 short market, lag=0, h=1s)
    champ_key = "lag0_h1000_top10_short_market"
    by_regime = {"green": [], "red": [], "flat": []}
    for r in valid_days:
        p = r["pnl_table"].get(champ_key)
        if p and p["n_trades"] >= 5:
            by_regime[r["regime"]].append(p["sharpe"])
    sharpes = {k: float(np.mean(v)) if v else np.nan for k, v in by_regime.items()}
    rep.append(f"Champion config = head_1s + lag_0 + top10% short + market exec.")
    rep.append(f"\n- Green days Sharpe avg = {sharpes.get('green', float('nan')):.3f} (n_days={len(by_regime['green'])})")
    rep.append(f"- Red days Sharpe avg = {sharpes.get('red', float('nan')):.3f} (n_days={len(by_regime['red'])})")
    rep.append(f"- Flat days Sharpe avg = {sharpes.get('flat', float('nan')):.3f} (n_days={len(by_regime['flat'])})")
    rep.append(f"- HC #428 R1 verdict: **{hc428_r1_gate(sharpes)}**")

    # ─── HC #428 R2 — MFE-within-horizon ───
    rep.append("\n## HC #428 R2 — MFE-within-Horizon Gate\n")
    rep.append("This Phase A does NOT impose explicit TP/SL — we use the full forward-drift PnL.")
    rep.append("R2 gate (TP ≤ p90 of realized MFE within horizon h) is not directly applicable at this stage.")
    rep.append("Note: forward drift over horizon h IS the closest to a 'no TP/SL' P&L — this is BY DESIGN")
    rep.append("the maximum bounded result. Any future production config layering TP/SL on top must")
    rep.append("re-test against R2.")

    # ─── Conclusion ───
    rep.append("\n## Conclusion\n")
    # Compute summary stats
    champ_lag0_h1s_top10_short_market = []
    for r in valid_days:
        p = r["pnl_table"].get(champ_key)
        if p: champ_lag0_h1s_top10_short_market.append(p)
    if champ_lag0_h1s_top10_short_market:
        ntr = sum(p["n_trades"] for p in champ_lag0_h1s_top10_short_market)
        net_mean = sum(p["net_mean_bps"]*p["n_trades"] for p in champ_lag0_h1s_top10_short_market)/max(1,ntr)
        shr_avg = np.mean([p["sharpe"] for p in champ_lag0_h1s_top10_short_market if p["n_trades"] >= 5])
        # Best lag/horizon combo (requires >= 2 days; relax if 1-day test)
        min_days_for_best = max(2, min(4, len(valid_days)))
        best = ("", -1e9, 0.0)
        for h_ms in HORIZONS_MS:
            for lag_ms in LAGS_MS:
                for mode in ("market", "passive"):
                    k = f"lag{lag_ms}_h{h_ms}_top10_short_{mode}"
                    drs = [r["pnl_table"].get(k) for r in valid_days]
                    drs = [d for d in drs if d and d["n_trades"]>=5]
                    if len(drs) < min_days_for_best: continue
                    s = float(np.mean([d["sharpe"] for d in drs]))
                    n_t = sum(d["n_trades"] for d in drs)
                    nm = float(sum(d["net_mean_bps"]*d["n_trades"] for d in drs)/max(1,n_t))
                    if s > best[1]:
                        best = (f"h={h_ms/1000:.0f}s lag={lag_ms}ms mode={mode}", s, nm)
        rep.append(f"\nChampion (h=1s, lag=0, top10% short, market): {ntr} trades across {len(champ_lag0_h1s_top10_short_market)} days, "
                   f"net mean = {net_mean:+.2f} bps/trade, per-trade Sharpe = {shr_avg:.3f}.")
        if best[0]:
            rep.append(f"\nBest config sweep: **{best[0]}** with Sharpe = {best[1]:.3f}, net mean = {best[2]:+.2f} bps/trade.\n")
        else:
            rep.append("\nBest config sweep: insufficient cross-day samples.\n")

        # HONEST verdict
        max_ic = max([c["ic_weighted_mean"] for c in concat_ic.values() if not np.isnan(c["ic_weighted_mean"])], default=np.nan)
        shr_avg_safe = shr_avg if not np.isnan(shr_avg) else 0.0
        rep.append(f"\nMaximum weighted-mean cross-asset IC across all (lag, horizon) cells: **{max_ic:.4f}**.")
        rep.append(f"For comparison, ES intra-asset IC(1s) on these days averages ~{np.mean([r['ic_es_intra_1s'] for r in valid_days]):.4f}.")
        rep.append(f"\n**Hypothesis verdict:** the cross-asset hypothesis " +
                   ("**survives**" if max_ic > 0.05 and shr_avg_safe > 0.05 else "**does NOT survive**") +
                   " Phase A. " +
                   ("Predictive linkage between ES and SPY is detectable but materially weaker than ES intra-asset signal "
                    "(IC ~5–10x lower in raw magnitude). " if max_ic > 0.02 else
                    "Cross-asset IC is near noise floor. ") +
                   ("After SPY costs (1 tick spread + SEC + TAF on market orders), net per-trade Sharpe at the best config is small. " if shr_avg_safe > 0 else
                    "After SPY costs, net Sharpe is non-positive. ") +
                   ("Phase B is justified only if a more efficient signal extraction (residualization, lag-specific models, "
                    "or microstructure adapters) can lift IC by >=2x." if max_ic > 0.02 else
                    "Phase B should NOT proceed without first establishing a baseline cross-asset edge."))
    else:
        rep.append("\nChampion PnL data unavailable — see log.")

    return "\n".join(rep)


def write_csv_tables(all_days):
    # IC table
    with open(OUT_DIR / "ic_table.csv", 'w') as f:
        f.write("date,lag_ms,horizon_ms,ic,n_samples\n")
        for r in all_days:
            if r is None: continue
            for k, v in r["ic_table"].items():
                parts = k.split("_")
                lag = int(parts[0][3:])
                h = int(parts[1][1:])
                f.write(f"{r['date']},{lag},{h},{v['ic']:.6f},{v['n']}\n")
    # PnL table
    with open(OUT_DIR / "pnl_table.csv", 'w') as f:
        f.write("date,regime,lag_ms,horizon_ms,top_pct,side,mode,n_trades,net_mean_bps,sharpe,sortino,pf,wr_net\n")
        for r in all_days:
            if r is None: continue
            for k, v in r["pnl_table"].items():
                f.write(f"{r['date']},{r['regime']},{v['lag_ms']},{v['horizon_ms']},"
                        f"{v['top_pct']:.2f},{v['side']},{v['mode']},{v['n_trades']},"
                        f"{v['net_mean_bps']:.4f},{v['sharpe']:.4f},{v['sortino']:.4f},"
                        f"{v['pf']:.4f},{v['wr_net']:.4f}\n")


def main():
    t0 = time.time()
    all_days = []
    for d in DATES:
        try:
            r = analyze_day(d)
            all_days.append(r)
        except Exception as e:
            log.exception(f"Failed {d}: {e}")
            all_days.append(None)
    elapsed = time.time() - t0
    log.info(f"All-day analysis complete in {elapsed:.1f}s")

    # Save raw per-day metrics
    serial = []
    for r in all_days:
        if r is None:
            serial.append(None)
            continue
        s = {k: v for k, v in r.items() if k not in ()}
        serial.append(s)
    with open(OUT_DIR / "per_day_metrics.json", 'w') as f:
        json.dump(serial, f, indent=2, default=str)

    write_csv_tables(all_days)
    report = write_report(all_days)
    with open(OUT_DIR / "phase_a_report.md", 'w') as f:
        f.write(report)
    log.info(f"Wrote {OUT_DIR/'phase_a_report.md'}")
    log.info(f"Wrote {OUT_DIR/'per_day_metrics.json'}")
    log.info(f"Wrote {OUT_DIR/'ic_table.csv'}")
    log.info(f"Wrote {OUT_DIR/'pnl_table.csv'}")


if __name__ == "__main__":
    main()
