#!/usr/bin/env python3
"""xasset_exec_grid_v2_hc541.py — Cross-asset (ES->SPY) grid v2.

Per HC #541: CORRECTED retail SPY cost model + LONGER horizons.

CHANGES FROM v1
===============
1. Cost model corrected (per HC #541 R1):
   - SPY passive RT: 0.3 bps  (SEC 0.23 + TAF 0.025, no spread crossing)
   - SPY market  RT: 1.8 bps  (passive + 1.5 bps spread crossing on $0.01 / $668)
   - v1 was effectively ~3-6 bps RT due to old SEC/TAF + bad spread cross.
2. Drop sub-second horizons. New menu: 5s / 30s / 2min / 5min / 15min.
3. Cancel window scales (~ half of hold) — not directly used here as we report
   pure hold-window drift; cancel window is documented as guidance for live.
4. TP/SL bounded by MFE-within-horizon — for the GRID we report raw hold drift
   (no TP/SL hits); MFE bound is implicit because we use the model's matching
   horizon head for confidence.
5. Sides: long-only, short-only, both. Quantiles: 1/5/10/20/50%.
6. Lags: 0 / 50 / 100 / 200 ms.

GATES (HC #428 R1 / #344)
=========================
- Per-day Sharpe > 1.5
- Per-day PF     > 1.4
- Per-day WR     > 0.55
- Regime gap     <= 0.5
- Day conc       <= 0.7
- n_days_active  >= 30  (will likely NOT pass — SPY mid-grid currently spans
                         only 9 days Mar 2-12 2026; reported regardless)

OUTPUTS
=======
/home/jupiter/Lvl3Quant/output/xasset_exec_grid_v2_hc541/
  results_table.csv      — all cells, all metrics, gate verdicts
  per_day_breakdown.csv  — top 50 cells x per-day
  survivors.txt          — passing cells (or "0 survivors")
  cost_comparison.txt    — top-5 cells: old (6 bps) vs new (1.8/0.3 bps) net
  run_log.txt
MLflow: experiment=xasset_exec_grid_v2_hc541
"""
from __future__ import annotations
import sys, time, logging, csv
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
ES_PRED_DIR  = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
ES_MBO_DIR   = ROOT / "data/processed/mbo_events_smart_v3"
SPY_GRID_DIR = ROOT / "data/processed/spy_mid_grid"
OUT_DIR      = ROOT / "output/xasset_exec_grid_v2_hc541"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use every date for which BOTH ES preds and SPY mid-grid exist.
def _discover_dates() -> List[str]:
    spy_dates = {p.name.split("_")[0] for p in SPY_GRID_DIR.glob("*_mid_250ms.npz")}
    es_dates  = {p.name.replace("oot_", "").replace(".npz", "")
                 for p in ES_PRED_DIR.glob("oot_*.npz")}
    return sorted(spy_dates & es_dates)

WINDOW = 1000
STRIDE = 250

# Grid axes
LAGS_MS         = [0, 50, 100, 200]
HORIZONS_MS     = [5000, 30000, 120000, 300000, 900000]  # 5s, 30s, 2m, 5m, 15m
CONF_QUANTILES  = [0.01, 0.05, 0.10, 0.20, 0.50]
SIDES           = ["short", "long", "both"]
MODES           = ["market", "passive"]

# Map horizon (ms) -> available v3.4.2 pred head used for confidence.
HEAD_FOR_HORIZON = {
    5000:   "pred_log_ret_5s",
    30000:  "pred_log_ret_30s",
    120000: "pred_log_ret_60s",      # nearest available head to 2min
    300000: "pred_log_ret_5min",
    900000: "pred_log_ret_5min",     # nearest available head to 15min
}

# --- Corrected SPY cost (HC #541 R1) ---
SPY_PASSIVE_RT_BPS = 0.3   # SEC 0.23 + TAF 0.025 (+epsilon); no spread cross
SPY_MARKET_RT_BPS  = 1.8   # passive + 1.5 bps spread crossing
PASSIVE_FILL_PROB  = 0.50

# Old (v1) cost approximation for cost_comparison.txt
OLD_PASSIVE_RT_BPS = 0.3 * 2  # ~0.6 bps (sec/taf x2 mis-stated in some configs)
OLD_MARKET_RT_BPS  = 6.0      # ~6 bps RT as cited in HC #541 description

# Gates
PER_DAY_SHARPE_MIN = 1.5
PER_DAY_PF_MIN     = 1.4
PER_DAY_WR_MIN     = 0.55
REGIME_GAP_REJECT  = 0.50
DAY_CONC_CAP       = 0.70
N_DAYS_MIN         = 30
REGIME_DAY_PCT_THRESHOLD = 0.10  # |close-open|% < 0.10% = flat

LOG_PATH = OUT_DIR / "run_log.txt"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(LOG_PATH, mode='w'),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


# =========================================================================
# Loaders
# =========================================================================

def load_es_preds(date_str: str) -> Optional[Dict]:
    es_path = ES_MBO_DIR / f"{date_str}_mbo_events.npz"
    pred_path = ES_PRED_DIR / f"oot_{date_str}.npz"
    if not es_path.exists() or not pred_path.exists():
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
    out = {"pred_ts_ns": pred_ts_ns, "n_pred": n_pred}
    for h_ms, head in HEAD_FOR_HORIZON.items():
        if head in pf.files:
            out[head] = pf[head][:n_pred]
    return out


def load_spy_grid(date_str: str) -> Optional[Dict]:
    p = SPY_GRID_DIR / f"{date_str}_mid_250ms.npz"
    if not p.exists():
        return None
    d = np.load(str(p), allow_pickle=True)
    return {
        "grid_ts_ns": d["grid_ts_ns"],
        "mid": d["mid_price"].astype(np.float64),
    }


def spy_at_time(grid_ts: np.ndarray, mid: np.ndarray, query_ns: np.ndarray) -> np.ndarray:
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
    h_ns   = int(horizon_ms) * 1_000_000
    entry_ts = pred_ts_ns + lag_ns
    exit_ts  = pred_ts_ns + lag_ns + h_ns
    mid_entry = spy_at_time(spy_grid["grid_ts_ns"], spy_grid["mid"], entry_ts)
    mid_exit  = spy_at_time(spy_grid["grid_ts_ns"], spy_grid["mid"], exit_ts)
    with np.errstate(invalid='ignore', divide='ignore'):
        drift_bps = (mid_exit - mid_entry) / mid_entry * 1e4
    return drift_bps, mid_entry


def classify_regime(spy_grid: Dict) -> Tuple[str, float]:
    m = spy_grid["mid"][spy_grid["mid"] > 0]
    if len(m) < 100:
        return "unknown", 0.0
    pct = (float(m[-1]) - float(m[0])) / float(m[0]) * 100.0
    if pct > REGIME_DAY_PCT_THRESHOLD:
        return "green", pct
    if pct < -REGIME_DAY_PCT_THRESHOLD:
        return "red", pct
    return "flat", pct


# =========================================================================
# Cost model (HC #541 R1)
# =========================================================================

def cost_bps(mode: str, n: int, passive_rt=SPY_PASSIVE_RT_BPS,
             market_rt=SPY_MARKET_RT_BPS) -> np.ndarray:
    rt = market_rt if mode == "market" else passive_rt
    return np.full(n, rt, dtype=np.float64)


# =========================================================================
# Metrics
# =========================================================================

def safe_sharpe(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2: return float('nan')
    sd = float(x.std(ddof=1))
    if sd <= 0: return float('nan')
    return float(x.mean() / sd)

def safe_sortino(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2: return float('nan')
    downside = x[x < 0]
    if len(downside) == 0:
        return float('inf') if x.mean() > 0 else 0.0
    dd = float(downside.std(ddof=1)) if len(downside) > 1 else float(abs(downside[0]))
    if dd <= 0: return float('nan')
    return float(x.mean() / dd)

def safe_pf(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    pos = float(x[x > 0].sum())
    neg = float(-x[x < 0].sum())
    if neg <= 0:
        return float('inf') if pos > 0 else 0.0
    return pos / neg

def max_dd_bps(x: np.ndarray) -> float:
    if len(x) == 0: return 0.0
    eq = np.cumsum(x)
    peak = np.maximum.accumulate(eq)
    return float((eq - peak).min())


# =========================================================================
# Per-day builder
# =========================================================================

def build_day_data(date_str: str) -> Optional[Dict]:
    es = load_es_preds(date_str)
    spy = load_spy_grid(date_str)
    if es is None or spy is None:
        return None
    regime, day_pct = classify_regime(spy)
    spy_ts_min = int(spy["grid_ts_ns"][0])
    spy_ts_max = int(spy["grid_ts_ns"][-1])
    max_exit_offset = (max(LAGS_MS) + max(HORIZONS_MS)) * 1_000_000
    in_win = (es["pred_ts_ns"] >= spy_ts_min) & (es["pred_ts_ns"] + max_exit_offset <= spy_ts_max)
    n_in = int(in_win.sum())
    if n_in < 100:
        log.warning(f"  {date_str}: only {n_in} preds fit grid window — skipping")
        return None
    pred_ts = es["pred_ts_ns"][in_win]
    head_pred = {}
    for h_ms, head in HEAD_FOR_HORIZON.items():
        if head in es:
            head_pred[h_ms] = es[head][in_win]
    drift = {}
    mid_entry = {}
    for lag_ms in LAGS_MS:
        for h_ms in HORIZONS_MS:
            d_bps, me = compute_spy_drift_bps(spy, pred_ts, lag_ms, h_ms)
            drift[(lag_ms, h_ms)] = d_bps
            mid_entry[(lag_ms, h_ms)] = me
    m_valid = spy["mid"][spy["mid"]>0]
    return {
        "date": date_str, "regime": regime, "day_pct": day_pct,
        "n_preds": n_in, "pred_ts_ns": pred_ts,
        "head_pred": head_pred, "drift_bps": drift, "mid_entry": mid_entry,
        "spy_open": float(m_valid[0]), "spy_close": float(m_valid[-1]),
    }


# =========================================================================
# Grid eval
# =========================================================================

def evaluate_cell(days: List[Dict], lag_ms: int, h_ms: int, q: float,
                  side: str, mode: str,
                  passive_rt: float = SPY_PASSIVE_RT_BPS,
                  market_rt: float = SPY_MARKET_RT_BPS) -> Optional[Dict]:
    pooled_net = []
    pooled_gross = []
    per_day_rows = []
    for d in days:
        if h_ms not in d["head_pred"]:
            continue
        pred = d["head_pred"][h_ms]
        drift = d["drift_bps"][(lag_ms, h_ms)]
        me = d["mid_entry"][(lag_ms, h_ms)]
        ok = np.isfinite(pred) & np.isfinite(drift) & np.isfinite(me) & (me > 0)
        if ok.sum() < 5:
            continue
        p_ok = pred[ok]; d_ok = drift[ok]
        thr = np.percentile(np.abs(p_ok), 100.0 * (1.0 - q))
        sel = np.abs(p_ok) >= thr
        if side == 'short':
            sel = sel & (p_ok < 0)
            sign = np.full(int(sel.sum()), -1.0)
        elif side == 'long':
            sel = sel & (p_ok > 0)
            sign = np.full(int(sel.sum()), +1.0)
        else:
            sign = np.sign(p_ok[sel])
            sign[sign == 0] = 1.0
        if sel.sum() < 3:
            continue
        gross_bps = d_ok[sel] * sign
        cb = cost_bps(mode, len(gross_bps), passive_rt, market_rt)
        if mode == 'passive':
            gross_bps = gross_bps * PASSIVE_FILL_PROB
            cb = cb * PASSIVE_FILL_PROB
        net_bps = gross_bps - cb
        pooled_net.append(net_bps)
        pooled_gross.append(gross_bps)
        per_day_rows.append({
            "date": d["date"], "regime": d["regime"], "day_pct": d["day_pct"],
            "n_trades": int(sel.sum()),
            "gross_sum_bps": float(gross_bps.sum()),
            "net_sum_bps": float(net_bps.sum()),
            "net_mean_bps": float(net_bps.mean()),
            "net_sharpe": safe_sharpe(net_bps),
            "net_pf": safe_pf(net_bps),
            "wr_net": float((net_bps > 0).mean()),
        })
    if not pooled_net:
        return None
    net = np.concatenate(pooled_net)
    gross = np.concatenate(pooled_gross)
    if len(net) < 10:
        return None
    # Per-day aggregate (Sharpe / PF / WR on per-day NET-mean series)
    daily_nm = np.array([r["net_mean_bps"] for r in per_day_rows], dtype=np.float64)
    daily_sum = np.array([r["net_sum_bps"] for r in per_day_rows], dtype=np.float64)
    cell = {
        "lag_ms": lag_ms, "horizon_ms": h_ms, "conf_q": q,
        "side": side, "mode": mode, "n_trades": int(len(net)),
        "n_days_active": len(per_day_rows),
        # pooled-trade-level metrics
        "sharpe_net": safe_sharpe(net),
        "sortino_net": safe_sortino(net),
        "pf_net": safe_pf(net),
        "wr_net": float((net > 0).mean()),
        "net_mean_bps": float(net.mean()),
        "net_median_bps": float(np.median(net)),
        "net_p10_bps": float(np.percentile(net, 10)),
        "net_p90_bps": float(np.percentile(net, 90)),
        "net_std_bps": float(net.std(ddof=1)),
        "gross_mean_bps": float(gross.mean()),
        "sharpe_gross": safe_sharpe(gross),
        "max_dd_bps": max_dd_bps(net),
        # per-day metrics (used by gates per HC #428)
        "per_day_sharpe":  safe_sharpe(daily_nm),
        "per_day_pf":      safe_pf(daily_sum),
        "per_day_wr":      float((daily_sum > 0).mean()),
        "per_day_mean_bps": float(daily_nm.mean()),
        # $ per day on $100k notional
        "dollars_per_day": float(daily_sum.mean() * 1e-4 * 100_000),
        "per_day": per_day_rows,
    }
    return cell


# =========================================================================
# Gates
# =========================================================================

def regime_gate(per_day: List[Dict]) -> Tuple[str, Dict]:
    by_regime = {"green": [], "red": [], "flat": []}
    for r in per_day:
        if r["regime"] in by_regime:
            by_regime[r["regime"]].append(r)
    info = {"n_green_days": len(by_regime["green"]),
            "n_red_days":   len(by_regime["red"]),
            "n_flat_days":  len(by_regime["flat"])}
    def regime_shr(rs):
        if not rs: return float('nan')
        v = np.array([r["net_sharpe"] for r in rs if np.isfinite(r["net_sharpe"])])
        if len(v) == 0: return float('nan')
        return float(v.mean())
    sg, sr = regime_shr(by_regime["green"]), regime_shr(by_regime["red"])
    info["sharpe_green"] = sg
    info["sharpe_red"]   = sr
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
    abs_sums = np.array([abs(r["net_sum_bps"]) for r in per_day])
    total = abs_sums.sum()
    if total <= 0:
        return False, 0.0
    max_share = float(abs_sums.max() / total)
    return max_share <= DAY_CONC_CAP, max_share


# =========================================================================
# Main
# =========================================================================

def main():
    t0 = time.time()
    DATES = _discover_dates()
    log.info("=" * 70)
    log.info("xasset_exec_grid_v2_hc541 — corrected SPY cost + longer horizons")
    log.info(f"Dates ({len(DATES)}): {DATES}")
    log.info(f"Lags ms: {LAGS_MS}")
    log.info(f"Horizons ms: {HORIZONS_MS}")
    log.info(f"Conf quantiles: {CONF_QUANTILES}")
    log.info(f"Sides: {SIDES}  Modes: {MODES}")
    log.info(f"Costs (RT bps): passive={SPY_PASSIVE_RT_BPS}  market={SPY_MARKET_RT_BPS}")
    log.info(f"Passive fill prob: {PASSIVE_FILL_PROB}")
    log.info("=" * 70)

    days = []
    for d_str in DATES:
        dd = build_day_data(d_str)
        if dd is None:
            log.warning(f"  {d_str}: SKIPPED")
            continue
        log.info(f"  {d_str}: regime={dd['regime']} day_pct={dd['day_pct']:+.3f}% "
                 f"n_preds={dd['n_preds']} spy=[{dd['spy_open']:.2f}->{dd['spy_close']:.2f}]")
        days.append(dd)
    if not days:
        log.error("No valid days. Aborting.")
        return
    log.info(f"Loaded {len(days)} days; total preds = {sum(d['n_preds'] for d in days):,}")

    if len(days) < N_DAYS_MIN:
        log.warning("DATA BLOCKER: only %d days available; HC #428 R1 requires "
                    "n_days>=%d. All cells will be flagged INSUFFICIENT_NDAYS but "
                    "results are still reported for diagnostic value.",
                    len(days), N_DAYS_MIN)

    # --- Grid eval ---
    log.info("Evaluating execution grid...")
    cells = []
    total = len(LAGS_MS) * len(HORIZONS_MS) * len(CONF_QUANTILES) * len(SIDES) * len(MODES)
    done = 0
    log_every = max(1, total // 10)
    for lag_ms in LAGS_MS:
        for h_ms in HORIZONS_MS:
            for q in CONF_QUANTILES:
                for side in SIDES:
                    for mode in MODES:
                        done += 1
                        cell = evaluate_cell(days, lag_ms, h_ms, q, side, mode)
                        if cell is None:
                            continue
                        rv, ri = regime_gate(cell["per_day"])
                        ok_dc, max_share = day_conc_check(cell["per_day"])
                        cell.update({
                            "regime_verdict": rv,
                            "sharpe_green": ri.get("sharpe_green", float('nan')),
                            "sharpe_red":   ri.get("sharpe_red",   float('nan')),
                            "regime_gap":   ri.get("regime_gap",   float('nan')),
                            "n_green_days": ri.get("n_green_days", 0),
                            "n_red_days":   ri.get("n_red_days",   0),
                            "n_flat_days":  ri.get("n_flat_days",  0),
                            "day_conc_max_share": max_share,
                            "day_conc_pass": ok_dc,
                        })
                        survives = (
                            cell["regime_verdict"] == "PASS_REGIME"
                            and ok_dc
                            and cell["per_day_sharpe"] > PER_DAY_SHARPE_MIN
                            and cell["per_day_pf"]     > PER_DAY_PF_MIN
                            and cell["per_day_wr"]     > PER_DAY_WR_MIN
                            and cell["n_days_active"] >= N_DAYS_MIN
                        )
                        cell["survives_all_gates"] = bool(survives)
                        cells.append(cell)
                        if done % log_every == 0:
                            log.info(f"  progress: {done}/{total}")
    log.info(f"Total cells with sufficient trades: {len(cells)}")

    # Sort by per_day_sharpe desc
    cells_sorted = sorted(
        cells,
        key=lambda c: (c["per_day_sharpe"] if np.isfinite(c["per_day_sharpe"]) else -1e9),
        reverse=True,
    )

    # --- results_table.csv ---
    fieldnames = [
        "lag_ms","horizon_ms","conf_q","side","mode",
        "n_trades","n_days_active",
        "per_day_sharpe","per_day_pf","per_day_wr","per_day_mean_bps","dollars_per_day",
        "sharpe_net","sortino_net","pf_net","wr_net",
        "net_mean_bps","net_median_bps","net_p10_bps","net_p90_bps","net_std_bps",
        "gross_mean_bps","sharpe_gross","max_dd_bps",
        "regime_verdict","sharpe_green","sharpe_red","regime_gap",
        "n_green_days","n_red_days","n_flat_days",
        "day_conc_max_share","day_conc_pass","survives_all_gates",
    ]
    with open(OUT_DIR / "results_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for c in cells_sorted:
            w.writerow({k: c.get(k) for k in fieldnames})
    log.info(f"Wrote results_table.csv ({len(cells_sorted)} rows)")

    # --- per_day_breakdown.csv (top 50) ---
    perday_rows = []
    for c in cells_sorted[:50]:
        for r in c["per_day"]:
            perday_rows.append({
                "lag_ms": c["lag_ms"], "horizon_ms": c["horizon_ms"], "conf_q": c["conf_q"],
                "side": c["side"], "mode": c["mode"],
                "date": r["date"], "regime": r["regime"], "day_pct": r["day_pct"],
                "n_trades": r["n_trades"], "gross_sum_bps": r["gross_sum_bps"],
                "net_sum_bps": r["net_sum_bps"], "net_mean_bps": r["net_mean_bps"],
                "net_sharpe": r["net_sharpe"], "net_pf": r["net_pf"], "wr_net": r["wr_net"],
            })
    if perday_rows:
        with open(OUT_DIR / "per_day_breakdown.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(perday_rows[0].keys()))
            w.writeheader(); w.writerows(perday_rows)
        log.info(f"Wrote per_day_breakdown.csv ({len(perday_rows)} rows)")

    # --- survivors.txt ---
    survivors = [c for c in cells_sorted if c["survives_all_gates"]]
    with open(OUT_DIR / "survivors.txt", "w") as f:
        if not survivors:
            f.write("0 survivors\n")
            f.write(f"(out of {len(cells_sorted)} cells evaluated)\n")
            f.write(f"n_days available = {len(days)} (gate requires >= {N_DAYS_MIN})\n")
            f.write("\nTop 10 by per-day Sharpe (FAILING gates):\n")
            f.write(f"  {'cell':<55}{'n':<7}{'days':<6}{'pdShr':<8}{'pdPF':<7}{'pdWR':<7}{'$/day':<10}{'reason'}\n")
            for c in cells_sorted[:10]:
                cell_id = f"lag{c['lag_ms']}_h{c['horizon_ms']}_q{int(c['conf_q']*100)}_{c['side']}_{c['mode']}"
                reasons = []
                if c["n_days_active"] < N_DAYS_MIN: reasons.append("ndays")
                if not (c["per_day_sharpe"] > PER_DAY_SHARPE_MIN): reasons.append("pdShr")
                if not (c["per_day_pf"]     > PER_DAY_PF_MIN):     reasons.append("pdPF")
                if not (c["per_day_wr"]     > PER_DAY_WR_MIN):     reasons.append("pdWR")
                if c["regime_verdict"] != "PASS_REGIME":           reasons.append(c["regime_verdict"])
                if not c["day_conc_pass"]:                         reasons.append("dayConc")
                f.write(f"  {cell_id:<55}{c['n_trades']:<7d}{c['n_days_active']:<6d}"
                        f"{c['per_day_sharpe']:<8.3f}{c['per_day_pf']:<7.2f}{c['per_day_wr']:<7.2%}"
                        f"{c['dollars_per_day']:<10.2f}{','.join(reasons)}\n")
        else:
            f.write(f"{len(survivors)} survivors\n\n")
            f.write(f"  {'cell':<55}{'n':<7}{'days':<6}{'pdShr':<8}{'pdPF':<7}{'pdWR':<7}{'$/day':<10}\n")
            for c in survivors:
                cell_id = f"lag{c['lag_ms']}_h{c['horizon_ms']}_q{int(c['conf_q']*100)}_{c['side']}_{c['mode']}"
                f.write(f"  {cell_id:<55}{c['n_trades']:<7d}{c['n_days_active']:<6d}"
                        f"{c['per_day_sharpe']:<8.3f}{c['per_day_pf']:<7.2f}{c['per_day_wr']:<7.2%}"
                        f"{c['dollars_per_day']:<10.2f}\n")
    log.info(f"Wrote survivors.txt ({len(survivors)} survivors)")

    # --- cost_comparison.txt — top 5 cells, old vs new cost ---
    top5 = cells_sorted[:5]
    with open(OUT_DIR / "cost_comparison.txt", "w") as f:
        f.write("Cost-correction impact on top 5 cells (sorted by per-day Sharpe).\n")
        f.write("Old cost: market RT 6.0 bps / passive RT 0.6 bps.\n")
        f.write("New cost: market RT 1.8 bps / passive RT 0.3 bps.\n\n")
        f.write(f"  {'cell':<50}{'mode':<9}{'old_net_mean':<14}{'new_net_mean':<14}"
                f"{'old_pdShr':<11}{'new_pdShr':<11}{'old_$/d':<10}{'new_$/d':<10}\n")
        for c in top5:
            old = evaluate_cell(days, c["lag_ms"], c["horizon_ms"], c["conf_q"],
                                c["side"], c["mode"],
                                passive_rt=OLD_PASSIVE_RT_BPS, market_rt=OLD_MARKET_RT_BPS)
            if old is None:
                continue
            cell_id = f"lag{c['lag_ms']}_h{c['horizon_ms']}_q{int(c['conf_q']*100)}_{c['side']}"
            f.write(f"  {cell_id:<50}{c['mode']:<9}"
                    f"{old['net_mean_bps']:<14.4f}{c['net_mean_bps']:<14.4f}"
                    f"{old['per_day_sharpe']:<11.3f}{c['per_day_sharpe']:<11.3f}"
                    f"{old['dollars_per_day']:<10.2f}{c['dollars_per_day']:<10.2f}\n")

        # Count how many cells would have passed at OLD cost vs NEW cost
        # using same gates (less ndays which is data-limited).
        def pass_sub(cell):
            return (cell.get("regime_verdict") == "PASS_REGIME"
                    and cell.get("day_conc_pass", False)
                    and cell.get("per_day_sharpe", 0) > PER_DAY_SHARPE_MIN
                    and cell.get("per_day_pf",    0) > PER_DAY_PF_MIN
                    and cell.get("per_day_wr",    0) > PER_DAY_WR_MIN)

        new_pass_sub = sum(1 for c in cells_sorted if pass_sub(c))

        # Re-eval all cells at OLD cost to count
        old_pass_sub = 0
        for c in cells_sorted:
            oc = evaluate_cell(days, c["lag_ms"], c["horizon_ms"], c["conf_q"],
                               c["side"], c["mode"],
                               passive_rt=OLD_PASSIVE_RT_BPS, market_rt=OLD_MARKET_RT_BPS)
            if oc is None: continue
            rv, _ = regime_gate(oc["per_day"])
            ok_dc, _ = day_conc_check(oc["per_day"])
            oc["regime_verdict"] = rv
            oc["day_conc_pass"]  = ok_dc
            if pass_sub(oc): old_pass_sub += 1
        f.write(f"\nGate-pass counts (ignoring n_days>={N_DAYS_MIN}; pure cost-driven):\n")
        f.write(f"  OLD cost: {old_pass_sub} cells pass (Shr/PF/WR + regime + day-conc)\n")
        f.write(f"  NEW cost: {new_pass_sub} cells pass (Shr/PF/WR + regime + day-conc)\n")
        f.write(f"  Delta from cost correction: +{new_pass_sub - old_pass_sub} additional passing cells.\n")

    # --- MLflow ---
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("xasset_exec_grid_v2_hc541")
        with mlflow.start_run(run_name="xasset_exec_grid_v2_hc541"):
            mlflow.log_param("dates", ",".join(DATES))
            mlflow.log_param("n_days", len(days))
            mlflow.log_param("lags_ms", LAGS_MS)
            mlflow.log_param("horizons_ms", HORIZONS_MS)
            mlflow.log_param("conf_quantiles", CONF_QUANTILES)
            mlflow.log_param("sides", SIDES)
            mlflow.log_param("modes", MODES)
            mlflow.log_param("spy_passive_rt_bps", SPY_PASSIVE_RT_BPS)
            mlflow.log_param("spy_market_rt_bps", SPY_MARKET_RT_BPS)
            mlflow.log_param("passive_fill_prob", PASSIVE_FILL_PROB)
            mlflow.log_metric("total_cells", len(cells_sorted))
            mlflow.log_metric("survivors", len(survivors))
            if cells_sorted:
                top = cells_sorted[0]
                mlflow.log_metric("best_per_day_sharpe", top["per_day_sharpe"])
                mlflow.log_metric("best_per_day_pf", top["per_day_pf"])
                mlflow.log_metric("best_per_day_wr", top["per_day_wr"])
                mlflow.log_metric("best_dollars_per_day", top["dollars_per_day"])
            for art in ["results_table.csv","per_day_breakdown.csv",
                        "survivors.txt","cost_comparison.txt","run_log.txt"]:
                p = OUT_DIR / art
                if p.exists(): mlflow.log_artifact(str(p))
        log.info("MLflow logging complete.")
    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")

    log.info(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
