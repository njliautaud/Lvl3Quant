#!/usr/bin/env python3
"""xasset_exec_grid_v1.py — Cross-asset (ES->SPY) execution grid.

Per HC #536: Use existing ES CNN-Mamba v3.4.2 predictions as the inference
source but EXECUTE trades on SPY shares (commission-free, $0.01 tick spread)
instead of ES futures.

PROTOCOL
========
For each of the matched-overlap RTH days (Mar 2,3,4,5,6,9 2026; Mar 8 missing
or holiday — we include whatever exists from {Mar 2..Mar 9}):

  1. Load v3.4.2 OOT predictions per date (from oot_47day_perdate).
  2. Map prediction i -> ES MBO event index 999 + i*250, look up timestamp.
  3. For each (signal_lag, hold_horizon) compute SPY mid at entry+lag and
     exit at entry+lag+horizon, drift in bps.
  4. Pool ALL trades across ALL days into one stream per grid cell.
  5. Per-grid-cell: Sharpe (per-trade), Sortino, PF, WR, n_trades, mean,
     median, p10, p90, MDD, per-day breakdown (incl. green/red regime).
  6. HC #428 R1 regime gate: reject if |Shr_green - Shr_red| / max(|...|) > 0.50.
  7. HC #344 day-conc cap: reject if any single day >= 70% of total |PnL|.
  8. Output sorted by net-PnL Sharpe descending.

GRID
====
signal_lag_ms     ∈ {0, 50, 100, 200, 500, 1000}
hold_horizon_ms   ∈ {1000, 5000, 30000}   # 1s, 5s, 30s per HC #428 R2
conf_quantile     ∈ {top1, top5, top10, top20, top50}
side              ∈ {short, long, both}
exec_mode         ∈ {market, passive}

Head used for confidence ranking = pred_log_ret_h where h matches horizon
(MFE-within-horizon, HC #428 R2).

COSTS (SPY at Alpaca/IBKR-Lite)
=================================
SPY tick = $0.01. Typical RTH spread = 1 tick.
- Market RT  : spread cross 1 full tick total + SEC fee on sell + TAF on sell.
                spread_bps = 1 cent / mid * 1e4 (~ 1.47 bps at SPY=$680).
- Passive RT : spread = 0 (sit on book both legs), but multiply gross drift by
                FILL_PROB (default 0.5). Fees still apply on filled sells.
- SEC fee    : $27.80 / $1,000,000 of SALES = 2.78e-5 * sell_notional
                = 0.278 bps on sell-side notional.
- TAF        : $0.000166 per share sold (we report ~ TAF_per_share / mid * 1e4 bps).
- No commission.

OUTPUTS
=======
/home/jupiter/Lvl3Quant/output/xasset_exec_grid_v1/results_table.csv
/home/jupiter/Lvl3Quant/output/xasset_exec_grid_v1/per_day_breakdown.csv
/home/jupiter/Lvl3Quant/output/xasset_exec_grid_v1/raw_ic_matrix.csv
/home/jupiter/Lvl3Quant/output/xasset_exec_grid_v1/run_log.txt
/home/jupiter/Lvl3Quant/output/xasset_exec_grid_v1/top10_survivors.txt

MLflow: experiment=xasset_exec_grid_v1
"""
from __future__ import annotations
import json, sys, time, logging, csv, os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
ES_PRED_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
ES_MBO_DIR  = ROOT / "data/processed/mbo_events_smart_v3"
SPY_GRID_DIR = ROOT / "data/processed/spy_mid_grid"
OUT_DIR = ROOT / "output/xasset_exec_grid_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Matched-overlap RTH days where BOTH ES MBO and SPY mid-grid exist.
# Mar 1 = Sunday. Mar 7-8 = weekend. So overlap = Mar 2,3,4,5,6,9.
DATES = ["20260302","20260303","20260304","20260305","20260306","20260309"]

WINDOW = 1000
STRIDE = 250

# Grid axes
LAGS_MS    = [0, 50, 100, 200, 500, 1000]
HORIZONS_MS = [1000, 5000, 30000]    # 1s, 5s, 30s (HC #428 R2 — short-horizon emphasized)
CONF_QUANTILES = [0.01, 0.05, 0.10, 0.20, 0.50]  # top X% by |pred|
SIDES = ["short", "long", "both"]
MODES = ["market", "passive"]

# Cost constants (SPY)
SPY_TICK_USD = 0.01
SPY_SEC_FEE_RATE = 27.8e-6           # $27.80 / $1M of sales notional
SPY_TAF_PER_SHARE = 1.66e-4          # $0.000166 per share sold (actually 1.66e-4, was wrong as 1.66e-5)
                                     # 2026 SIFMA TAF = $0.000166 per share => 1.66e-4
SPY_COMMISSION_PER_SH = 0.0
PASSIVE_FILL_PROB = 0.50

# Regime gate
REGIME_GAP_REJECT = 0.50  # HC #428 R1
DAY_CONC_CAP = 0.70       # HC #344
REGIME_DAY_PCT_THRESHOLD = 0.10  # SPY close-to-close % move; |move| < 0.10% = flat

LOG_PATH = OUT_DIR / "run_log.txt"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(LOG_PATH, mode='w'),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


# =========================================================================
# Loaders
# =========================================================================

def load_es_pred_times(date_str: str) -> Optional[Dict]:
    es_path = ES_MBO_DIR / f"{date_str}_mbo_events.npz"
    pred_path = ES_PRED_DIR / f"oot_{date_str}.npz"
    if not es_path.exists() or not pred_path.exists():
        log.warning(f"missing ES data for {date_str}: mbo={es_path.exists()} pred={pred_path.exists()}")
        return None
    es = np.load(str(es_path), allow_pickle=True)
    timestamps = es["timestamps"]
    pf = np.load(str(pred_path), allow_pickle=True)
    n_pred = pf["pred_log_ret_1s"].shape[0]
    idx = (WINDOW - 1) + np.arange(n_pred) * STRIDE
    if idx[-1] >= len(timestamps):
        ok_n = int(np.searchsorted(idx, len(timestamps), side='left'))
        idx = idx[:ok_n]
        n_pred = ok_n
    pred_ts_ns = timestamps[idx]
    out = {
        "pred_ts_ns": pred_ts_ns,
        "pred_log_ret_1s":  pf["pred_log_ret_1s"][:n_pred],
        "pred_log_ret_5s":  pf["pred_log_ret_5s"][:n_pred],
        "pred_log_ret_30s": pf["pred_log_ret_30s"][:n_pred],
        "tgt_log_ret_1s":   pf["target_log_ret_1s"][:n_pred],
        "mask_log_ret_1s":  pf["mask_log_ret_1s"][:n_pred],
    }
    return out


def load_spy_grid(date_str: str) -> Optional[Dict]:
    p = SPY_GRID_DIR / f"{date_str}_mid_250ms.npz"
    if not p.exists():
        return None
    d = np.load(str(p), allow_pickle=True)
    return {
        "grid_ts_ns": d["grid_ts_ns"],
        "mid": d["mid_price"].astype(np.float64),
        "bid": d["bid_price"].astype(np.float64),
        "ask": d["ask_price"].astype(np.float64),
        "spread_ticks": d["spread_ticks"].astype(np.float32),
    }


# =========================================================================
# Alignment + drift
# =========================================================================

def spy_at_time(grid_ts: np.ndarray, mid: np.ndarray, query_ns: np.ndarray) -> np.ndarray:
    """For each query, LAST grid point <= query."""
    idx = np.searchsorted(grid_ts, query_ns, side='right') - 1
    valid = (idx >= 0) & (idx < len(grid_ts))
    out = np.full(len(query_ns), np.nan, dtype=np.float64)
    if valid.any():
        cand = mid[idx[valid]]
        out[valid] = np.where(cand > 0, cand, np.nan)
    return out


def compute_spy_drift_bps(spy_grid: Dict, pred_ts_ns: np.ndarray,
                          lag_ms: int, horizon_ms: int) -> Tuple[np.ndarray, np.ndarray]:
    lag_ns = int(lag_ms) * 1_000_000
    h_ns = int(horizon_ms) * 1_000_000
    entry_ts = pred_ts_ns + lag_ns
    exit_ts = pred_ts_ns + lag_ns + h_ns
    mid_entry = spy_at_time(spy_grid["grid_ts_ns"], spy_grid["mid"], entry_ts)
    mid_exit  = spy_at_time(spy_grid["grid_ts_ns"], spy_grid["mid"], exit_ts)
    with np.errstate(invalid='ignore', divide='ignore'):
        drift_bps = (mid_exit - mid_entry) / mid_entry * 1e4
    return drift_bps, mid_entry


# =========================================================================
# Regime classification (SPY close-to-close)
# =========================================================================

def classify_regime(spy_grid: Dict, threshold_pct: float = REGIME_DAY_PCT_THRESHOLD) -> Tuple[str, float]:
    mid_valid = spy_grid["mid"][spy_grid["mid"] > 0]
    if len(mid_valid) < 100:
        return "unknown", 0.0
    open_p, close_p = float(mid_valid[0]), float(mid_valid[-1])
    pct = (close_p - open_p) / open_p * 100.0
    if pct > threshold_pct:
        return "green", pct
    if pct < -threshold_pct:
        return "red", pct
    return "flat", pct


# =========================================================================
# Cost model
# =========================================================================

def per_trade_costs_bps(mid_entry: np.ndarray, mode: str) -> np.ndarray:
    """Round-trip cost in bps of entry notional.
    market : 1 full tick spread crossing total + SEC + TAF on the sell leg.
    passive: 0 spread, SEC + TAF on the sell leg (we sit on the book both legs).
    For passive we additionally apply PASSIVE_FILL_PROB by scaling the gross
    drift in the caller (not here).
    """
    me = np.where(mid_entry > 0, mid_entry, np.nan)
    spread_bps = (SPY_TICK_USD / me) * 1e4 if mode == 'market' else np.zeros_like(me)
    # SEC fee on sell-side notional (~ same as entry notional to first order)
    sec_bps = SPY_SEC_FEE_RATE * 1e4  # = 0.278 bps (constant)
    # TAF per share -> bps of mid
    taf_bps = (SPY_TAF_PER_SHARE / me) * 1e4
    # commission = 0
    total = spread_bps + sec_bps + taf_bps
    return total


# =========================================================================
# Metrics
# =========================================================================

def safe_sharpe(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float('nan')
    sd = float(x.std(ddof=1))
    if sd <= 0:
        return float('nan')
    return float(x.mean() / sd)

def safe_sortino(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float('nan')
    downside = x[x < 0]
    if len(downside) == 0:
        return float('inf') if x.mean() > 0 else 0.0
    dd = float(downside.std(ddof=1)) if len(downside) > 1 else float(abs(downside[0]))
    if dd <= 0:
        return float('nan')
    return float(x.mean() / dd)

def safe_pf(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    pos = float(x[x > 0].sum())
    neg = float(-x[x < 0].sum())
    if neg <= 0:
        return float('inf') if pos > 0 else 0.0
    return pos / neg

def max_dd_bps(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    eq = np.cumsum(x)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak  # negative
    return float(dd.min())

def pearson_ic(pred: np.ndarray, target: np.ndarray) -> Tuple[float, int]:
    m = np.isfinite(pred) & np.isfinite(target)
    if m.sum() < 100:
        return float('nan'), int(m.sum())
    p = pred[m].astype(np.float64); t = target[m].astype(np.float64)
    if p.std() == 0 or t.std() == 0:
        return float('nan'), int(m.sum())
    return float(np.corrcoef(p, t)[0,1]), int(m.sum())


# =========================================================================
# Per-day data builder
# =========================================================================

def build_day_data(date_str: str) -> Optional[Dict]:
    """Returns dict with per-day arrays:
      pred_ts_ns, mid_entry_by_lag[lag], drift_bps_by_lag_h[(lag,h)],
      head_pred[h], regime, day_pct."""
    es = load_es_pred_times(date_str)
    spy = load_spy_grid(date_str)
    if es is None or spy is None:
        return None
    regime, day_pct = classify_regime(spy)
    # Filter to ES predictions within the SPY RTH window (so lookups don't NaN)
    spy_ts_min, spy_ts_max = int(spy["grid_ts_ns"][0]), int(spy["grid_ts_ns"][-1])
    # Need pred_ts + max(lag) + max(horizon) <= spy_ts_max
    max_exit_offset = (max(LAGS_MS) + max(HORIZONS_MS)) * 1_000_000
    in_win = (es["pred_ts_ns"] >= spy_ts_min) & (es["pred_ts_ns"] + max_exit_offset <= spy_ts_max)
    n_in = int(in_win.sum())
    if n_in < 100:
        log.warning(f"  {date_str}: only {n_in} preds fit grid window — skipping")
        return None
    pred_ts = es["pred_ts_ns"][in_win]
    head_pred = {
        1000:  es["pred_log_ret_1s"][in_win],
        5000:  es["pred_log_ret_5s"][in_win],
        30000: es["pred_log_ret_30s"][in_win],
    }
    # Precompute SPY drift + mid_entry for every (lag, h) pair
    drift = {}
    mid_entry = {}
    for lag_ms in LAGS_MS:
        for h_ms in HORIZONS_MS:
            d_bps, me = compute_spy_drift_bps(spy, pred_ts, lag_ms, h_ms)
            drift[(lag_ms, h_ms)] = d_bps
            mid_entry[(lag_ms, h_ms)] = me
    return {
        "date": date_str,
        "regime": regime,
        "day_pct": day_pct,
        "n_preds": n_in,
        "pred_ts_ns": pred_ts,
        "head_pred": head_pred,
        "drift_bps": drift,
        "mid_entry": mid_entry,
        "spy_open": float(spy["mid"][spy["mid"]>0][0]),
        "spy_close": float(spy["mid"][spy["mid"]>0][-1]),
    }


# =========================================================================
# Grid PnL
# =========================================================================

def evaluate_cell(days: List[Dict], lag_ms: int, h_ms: int, q: float, side: str, mode: str) -> Optional[Dict]:
    """Pool all trades across days for this grid cell. Apply confidence
    threshold per-day (top-q% by |pred|), keep selected trades, compute net
    PnL, then aggregate metrics."""
    # Compute per-day threshold so confidence% is per-day (regime-fair).
    pooled_net = []
    pooled_gross = []
    per_day_rows = []  # for regime + day-conc check
    for d in days:
        pred = d["head_pred"][h_ms]
        drift = d["drift_bps"][(lag_ms, h_ms)]
        me = d["mid_entry"][(lag_ms, h_ms)]
        ok = np.isfinite(pred) & np.isfinite(drift) & np.isfinite(me) & (me > 0)
        if ok.sum() < 5:
            continue
        p_ok = pred[ok]
        d_ok = drift[ok]
        me_ok = me[ok]
        thr = np.percentile(np.abs(p_ok), 100.0 * (1.0 - q))
        sel = np.abs(p_ok) >= thr
        if side == 'short':
            sel = sel & (p_ok < 0)
            sign = np.full(sel.sum(), -1.0)
        elif side == 'long':
            sel = sel & (p_ok > 0)
            sign = np.full(sel.sum(), +1.0)
        else:
            sign = np.sign(p_ok[sel])
            sign[sign == 0] = 1.0
        if sel.sum() < 3:
            continue
        gross_bps = d_ok[sel] * sign
        cost_bps = per_trade_costs_bps(me_ok[sel], mode)
        if mode == 'passive':
            # Apply fill prob: with prob (1-p) the trade doesn't fill, contributing 0.
            # We model expectation: scale gross by fill_prob, cost only applies on filled portion.
            gross_bps = gross_bps * PASSIVE_FILL_PROB
            cost_bps = cost_bps * PASSIVE_FILL_PROB
        net_bps = gross_bps - cost_bps
        pooled_net.append(net_bps)
        pooled_gross.append(gross_bps)
        per_day_rows.append({
            "date": d["date"], "regime": d["regime"], "day_pct": d["day_pct"],
            "n_trades": int(sel.sum()),
            "gross_sum_bps": float(gross_bps.sum()),
            "net_sum_bps": float(net_bps.sum()),
            "net_mean_bps": float(net_bps.mean()),
            "net_sharpe": safe_sharpe(net_bps),
            "wr_net": float((net_bps > 0).mean()),
        })
    if not pooled_net:
        return None
    net = np.concatenate(pooled_net)
    gross = np.concatenate(pooled_gross)
    n = len(net)
    if n < 10:
        return None
    cell = {
        "lag_ms": lag_ms, "horizon_ms": h_ms, "conf_q": q,
        "side": side, "mode": mode, "n_trades": n,
        "gross_mean_bps": float(gross.mean()),
        "net_mean_bps": float(net.mean()),
        "net_median_bps": float(np.median(net)),
        "net_p10_bps": float(np.percentile(net, 10)),
        "net_p90_bps": float(np.percentile(net, 90)),
        "net_std_bps": float(net.std(ddof=1)),
        "sharpe_net": safe_sharpe(net),
        "sortino_net": safe_sortino(net),
        "pf_net": safe_pf(net),
        "sharpe_gross": safe_sharpe(gross),
        "wr_net": float((net > 0).mean()),
        "max_dd_bps": max_dd_bps(net),
        "n_days_active": len(per_day_rows),
        "per_day": per_day_rows,
    }
    return cell


# =========================================================================
# Gate checks
# =========================================================================

def regime_gate(per_day: List[Dict]) -> Tuple[str, Dict]:
    """HC #428 R1 — green vs red Sharpe gap."""
    by_regime = {"green": [], "red": [], "flat": []}
    for r in per_day:
        if r["regime"] in by_regime:
            by_regime[r["regime"]].append(r)
    info = {"n_green_days": len(by_regime["green"]),
            "n_red_days": len(by_regime["red"]),
            "n_flat_days": len(by_regime["flat"])}
    # Pool net PnL per regime to compute regime-level Sharpe? We have per-day Sharpes only here.
    # Better: weight-mean of daily Sharpes, but cleaner approach is to compute on pooled net.
    # We'll compute on per-day Sharpes (regime-level Sharpe approx).
    def regime_shr(rs):
        if not rs:
            return float('nan')
        v = np.array([r["net_sharpe"] for r in rs if np.isfinite(r["net_sharpe"])])
        if len(v) == 0:
            return float('nan')
        return float(v.mean())
    sg = regime_shr(by_regime["green"])
    sr = regime_shr(by_regime["red"])
    info["sharpe_green"] = sg
    info["sharpe_red"] = sr
    if not (np.isfinite(sg) and np.isfinite(sr)):
        info["verdict"] = "INSUFFICIENT_REGIME_COVERAGE"
        return info["verdict"], info
    denom = max(abs(sg), abs(sr))
    if denom == 0:
        info["verdict"] = "FLAT_ZERO"
        return info["verdict"], info
    gap = abs(sg - sr) / denom
    info["regime_gap"] = float(gap)
    info["verdict"] = "REJECT_REGIME" if gap > REGIME_GAP_REJECT else "PASS_REGIME"
    return info["verdict"], info


def day_conc_check(per_day: List[Dict]) -> Tuple[bool, float]:
    """HC #344 — no day > DAY_CONC_CAP of total |net PnL|."""
    abs_sums = np.array([abs(r["net_sum_bps"]) for r in per_day])
    total = abs_sums.sum()
    if total <= 0:
        return False, 0.0
    max_share = float(abs_sums.max() / total)
    return max_share <= DAY_CONC_CAP, max_share


def all_short_red_only_check(per_day: List[Dict], side: str) -> bool:
    """HC #428 R1 — reject all-short configs developed on red days only."""
    if side != 'short':
        return True  # pass
    regimes = set(r["regime"] for r in per_day if r["n_trades"] > 0)
    has_non_red = bool(regimes - {"red"})
    return has_non_red


# =========================================================================
# Raw IC matrix
# =========================================================================

def compute_ic_matrix(days: List[Dict]) -> Dict[Tuple[int,int], Tuple[float,int]]:
    """Pool predictions and SPY drifts across days, compute IC per (lag, h).
    Uses the head matching the horizon (HC #428 R2)."""
    result = {}
    for lag_ms in LAGS_MS:
        for h_ms in HORIZONS_MS:
            preds_all = []
            drifts_all = []
            for d in days:
                p = d["head_pred"][h_ms]
                dr = d["drift_bps"][(lag_ms, h_ms)]
                m = np.isfinite(p) & np.isfinite(dr)
                preds_all.append(p[m])
                drifts_all.append(dr[m])
            preds = np.concatenate(preds_all)
            drifts = np.concatenate(drifts_all)
            ic, n = pearson_ic(preds, drifts)
            result[(lag_ms, h_ms)] = (ic, n)
    return result


# =========================================================================
# Main
# =========================================================================

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("xasset_exec_grid_v1 — HC #536 cross-asset ES->SPY grid")
    log.info(f"Dates: {DATES}")
    log.info(f"Grid: lags={LAGS_MS}ms x horizons={HORIZONS_MS}ms x "
             f"conf_q={CONF_QUANTILES} x sides={SIDES} x modes={MODES}")
    log.info("=" * 70)

    # --- Load all day data ---
    days = []
    for d_str in DATES:
        log.info(f"Loading {d_str}...")
        dd = build_day_data(d_str)
        if dd is not None:
            log.info(f"  {d_str}: regime={dd['regime']} day_pct={dd['day_pct']:+.3f}% "
                     f"n_preds={dd['n_preds']} spy=[{dd['spy_open']:.2f}->{dd['spy_close']:.2f}]")
            days.append(dd)
        else:
            log.warning(f"  {d_str}: SKIPPED")
    if not days:
        log.error("No valid days. Aborting.")
        return
    log.info(f"Loaded {len(days)} days; total preds = {sum(d['n_preds'] for d in days):,}")

    # --- IC matrix (raw signal strength before exec costs) ---
    log.info("Computing raw IC matrix (head-horizon matched)...")
    ic_mat = compute_ic_matrix(days)
    ic_rows = []
    log.info("IC matrix (Pearson, ES pred head matches horizon):")
    log.info(f"  {'horizon':<10}{'lag_ms':<10}{'IC':<12}{'n':<12}")
    for h_ms in HORIZONS_MS:
        for lag_ms in LAGS_MS:
            ic, n = ic_mat[(lag_ms, h_ms)]
            log.info(f"  {h_ms/1000:<10.0f}{lag_ms:<10d}{ic:<12.5f}{n:<12d}")
            ic_rows.append({"horizon_s": h_ms/1000, "lag_ms": lag_ms, "ic": ic, "n": n})

    with open(OUT_DIR / "raw_ic_matrix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["horizon_s","lag_ms","ic","n"])
        w.writeheader(); w.writerows(ic_rows)

    # --- Grid eval ---
    log.info("Evaluating execution grid...")
    cells = []
    total_combos = len(LAGS_MS) * len(HORIZONS_MS) * len(CONF_QUANTILES) * len(SIDES) * len(MODES)
    done = 0
    for lag_ms in LAGS_MS:
        for h_ms in HORIZONS_MS:
            for q in CONF_QUANTILES:
                for side in SIDES:
                    for mode in MODES:
                        cell = evaluate_cell(days, lag_ms, h_ms, q, side, mode)
                        done += 1
                        if cell is None:
                            continue
                        # Apply gates
                        regime_verdict, regime_info = regime_gate(cell["per_day"])
                        ok_day_conc, max_share = day_conc_check(cell["per_day"])
                        ok_not_all_short_red = all_short_red_only_check(cell["per_day"], side)
                        cell.update({
                            "regime_verdict": regime_verdict,
                            "sharpe_green": regime_info.get("sharpe_green", float('nan')),
                            "sharpe_red":   regime_info.get("sharpe_red", float('nan')),
                            "regime_gap":   regime_info.get("regime_gap", float('nan')),
                            "n_green_days": regime_info.get("n_green_days", 0),
                            "n_red_days":   regime_info.get("n_red_days", 0),
                            "day_conc_max_share": max_share,
                            "day_conc_pass": ok_day_conc,
                            "not_all_short_red_pass": ok_not_all_short_red,
                        })
                        survives = (
                            cell["regime_verdict"] == "PASS_REGIME"
                            and ok_day_conc
                            and ok_not_all_short_red
                            and cell["sharpe_net"] > 1.0
                            and cell["pf_net"] > 1.2
                            and cell["n_trades"] >= 30
                        )
                        cell["survives_all_gates"] = bool(survives)
                        cells.append(cell)
        log.info(f"  progress: ~{done}/{total_combos} cells")

    log.info(f"Total cells with >=10 trades: {len(cells)}")

    # --- Sort by net Sharpe desc ---
    cells_sorted = sorted(
        cells,
        key=lambda c: (c["sharpe_net"] if np.isfinite(c["sharpe_net"]) else -1e9),
        reverse=True,
    )

    # --- Save results CSV ---
    fieldnames = [
        "lag_ms","horizon_ms","conf_q","side","mode","n_trades","n_days_active",
        "sharpe_net","sortino_net","pf_net","wr_net",
        "net_mean_bps","net_median_bps","net_p10_bps","net_p90_bps","net_std_bps",
        "gross_mean_bps","sharpe_gross","max_dd_bps",
        "regime_verdict","sharpe_green","sharpe_red","regime_gap",
        "n_green_days","n_red_days",
        "day_conc_max_share","day_conc_pass","not_all_short_red_pass",
        "survives_all_gates",
    ]
    with open(OUT_DIR / "results_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for c in cells_sorted:
            row = {k: c.get(k) for k in fieldnames}
            w.writerow(row)
    log.info(f"Wrote results_table.csv ({len(cells_sorted)} rows)")

    # --- Per-day breakdown CSV (for top-50 only, by sharpe) ---
    perday_rows = []
    for c in cells_sorted[:50]:
        for r in c["per_day"]:
            perday_rows.append({
                "lag_ms": c["lag_ms"], "horizon_ms": c["horizon_ms"], "conf_q": c["conf_q"],
                "side": c["side"], "mode": c["mode"],
                "date": r["date"], "regime": r["regime"], "day_pct": r["day_pct"],
                "n_trades": r["n_trades"], "gross_sum_bps": r["gross_sum_bps"],
                "net_sum_bps": r["net_sum_bps"], "net_mean_bps": r["net_mean_bps"],
                "net_sharpe": r["net_sharpe"], "wr_net": r["wr_net"],
            })
    if perday_rows:
        with open(OUT_DIR / "per_day_breakdown.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(perday_rows[0].keys()))
            w.writeheader(); w.writerows(perday_rows)

    # --- Survivors / top 10 ---
    survivors = [c for c in cells_sorted if c["survives_all_gates"]]
    top10 = cells_sorted[:10]
    log.info("=" * 70)
    log.info(f"SURVIVORS (Sharpe>1.0 AND PF>1.2 AND regime-pass AND day-conc-pass): {len(survivors)}")
    log.info("=" * 70)
    log.info("TOP 10 by net Sharpe:")
    log.info(f"  {'cell':<60}{'n':<7}{'Shr':<8}{'PF':<8}{'WR':<8}{'net_mean_bps':<14}{'gates'}")
    lines = []
    for c in top10:
        cell_id = f"lag{c['lag_ms']}_h{c['horizon_ms']}_q{int(c['conf_q']*100)}_{c['side']}_{c['mode']}"
        gate_str = "SURV" if c["survives_all_gates"] else c["regime_verdict"]
        line = (f"  {cell_id:<60}{c['n_trades']:<7d}"
                f"{c['sharpe_net']:<8.3f}{c['pf_net']:<8.2f}"
                f"{c['wr_net']:<8.2%}{c['net_mean_bps']:<14.3f}{gate_str}")
        log.info(line)
        lines.append(line)

    with open(OUT_DIR / "top10_survivors.txt", "w") as f:
        f.write("TOP 10 cells by net Sharpe:\n")
        f.write("\n".join(lines))
        f.write(f"\n\nSurvivors meeting all gates (Shr>1, PF>1.2, regime-pass, day-conc-pass): {len(survivors)}\n")
        for c in survivors[:20]:
            cell_id = f"lag{c['lag_ms']}_h{c['horizon_ms']}_q{int(c['conf_q']*100)}_{c['side']}_{c['mode']}"
            f.write(f"  {cell_id}  Shr={c['sharpe_net']:.3f} PF={c['pf_net']:.2f} n={c['n_trades']} "
                    f"net_mean={c['net_mean_bps']:.3f} bps regime_gap={c['regime_gap']:.2%}\n")

    # --- MLflow ---
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("xasset_exec_grid_v1")
        with mlflow.start_run(run_name=f"grid_{time.strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("dates", ",".join(DATES))
            mlflow.log_param("n_days", len(days))
            mlflow.log_param("lags_ms", LAGS_MS)
            mlflow.log_param("horizons_ms", HORIZONS_MS)
            mlflow.log_param("conf_quantiles", CONF_QUANTILES)
            mlflow.log_param("sides", SIDES)
            mlflow.log_param("modes", MODES)
            mlflow.log_param("passive_fill_prob", PASSIVE_FILL_PROB)
            mlflow.log_metric("total_cells", len(cells_sorted))
            mlflow.log_metric("survivors", len(survivors))
            if cells_sorted:
                mlflow.log_metric("best_sharpe_net", cells_sorted[0]["sharpe_net"])
                mlflow.log_metric("best_pf_net", cells_sorted[0]["pf_net"])
                mlflow.log_metric("best_net_mean_bps", cells_sorted[0]["net_mean_bps"])
            mlflow.log_artifact(str(OUT_DIR / "results_table.csv"))
            mlflow.log_artifact(str(OUT_DIR / "raw_ic_matrix.csv"))
            mlflow.log_artifact(str(OUT_DIR / "top10_survivors.txt"))
            if (OUT_DIR / "per_day_breakdown.csv").exists():
                mlflow.log_artifact(str(OUT_DIR / "per_day_breakdown.csv"))
        log.info("MLflow logging complete.")
    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")

    log.info(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
