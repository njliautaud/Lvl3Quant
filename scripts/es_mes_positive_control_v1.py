#!/usr/bin/env python3
"""es_mes_positive_control_v1.py — Native ES positive control + MES re-costing.

CONTEXT (HC #536 R4)
====================
Three parallel cross-asset experiments (xasset_exec_grid_v1, es_spy_basis_pair_v1,
es_spy_leadlag_v1) all returned DEAD on the SPY side, with the binding cause
identified as signal-transfer loss (ES->SPY raw-return correlation only ~0.15
at 50ms timescale, not the >0.99 daily correlation assumed).

This experiment is the positive-control pivot:

  PART A — Re-validate v3.4.2 signal executed on ES ITSELF on the same 6-day
           window (Mar 2,3,4,5,6,9 2026 RTH) using the cached predictions.
           If this is also DEAD, the signal itself decayed. If it surfaces
           profitable cells, the signal is alive and the cross-asset failure
           was purely transfer-loss.

  PART B — Re-cost the same surviving cells using MES (Micro E-mini, 1/10
           notional) economics. MES is the only contract that lets a retail
           account ($25-100k) size correctly without 10x notional concentration.

PROTOCOL
========
- Same window as xasset_exec_grid_v1: 20260302, 03, 04, 05, 06, 09 (RTH).
- Same v3.4.2 predictions: output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate/.
- Predictions are emitted at MBO event index (WINDOW-1) + i*STRIDE in
  data/processed/mbo_events_smart_v3/<DATE>_mbo_events.npz (WINDOW=1000, STRIDE=250).
- ES last-trade prices read from data/derived/mid_price_cache_hc439/<DATE>_trades.npz
  ('last trade <= ts' reduction used as ES mid proxy).

GRID
====
  horizon       in {1s, 5s, 30s}   (HC #428 R2 head-matched)
  conf_quantile in {top1%, top5%, top10%, top20%, top50%}
  side          in {short, long, both}
  exec_mode     in {market, passive}

PnL is computed on ES. Confidence ranking head matches horizon (HC #428 R2).

COSTS — ES (CANONICAL, CLAUDE.md)
================================
  ES tick value $12.50, RT commission $4.70 = 0.376 ticks
  Market RT  : 0.376 ticks comm + 1.0 tick spread = 1.376 ticks
  Passive RT : 0.376 ticks comm (50% fill prob -> halve gross + cost)
  Notional / contract ~$11,500/pt * 50 = ~$230k at SPX 4600
    1.376 ticks = $17.20 / $230k ~ 0.75 bps RT (market)
    0.376 ticks = $4.70 / $230k ~ 0.20 bps RT (passive)

COSTS — MES (RETAIL MICRO)
==========================
  MES tick value $1.25 (1/10 of ES)
  MES RT commission ~$1.50 (AMP/IBKR-Lite retail microE)
  Market RT  : $1.50 comm + $1.25 spread = $2.75 RT
  Passive RT : $1.50 comm = $1.50 RT
  Notional ~ $23k at SPX 4600
    Market : $2.75 / $23k ~ 1.20 bps RT
    Passive: $1.50 / $23k ~ 0.65 bps RT
  MES is ~1.6x more expensive in bps but 10x smaller notional.

GATES (HC #428 R1 + HC #344 + minimum bars)
===========================================
  - Regime gap: |Sharpe_green - Sharpe_red| / max(|...|) <= 0.50
  - Day concentration: no single day > 70% of total |net PnL|
  - No all-short-on-red-only
  - n_trades >= 30
  - Sharpe_net > 1.0
  - PF_net > 1.2

OUTPUTS
=======
  /home/jupiter/Lvl3Quant/output/es_mes_positive_control_v1/results_table.csv
  /home/jupiter/Lvl3Quant/output/es_mes_positive_control_v1/per_day_breakdown.csv
  /home/jupiter/Lvl3Quant/output/es_mes_positive_control_v1/survivors.txt
  /home/jupiter/Lvl3Quant/output/es_mes_positive_control_v1/run_log.txt

MLflow experiment: es_mes_positive_control_v1
"""
from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
ES_PRED_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
ES_MBO_DIR = ROOT / "data/processed/mbo_events_smart_v3"
ES_TRADE_DIR = ROOT / "data/derived/mid_price_cache_hc439"
OUT_DIR = ROOT / "output/es_mes_positive_control_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATES = ["20260302", "20260303", "20260304", "20260305", "20260306", "20260309"]

WINDOW = 1000
STRIDE = 250

HORIZONS_MS = [1000, 5000, 30000]            # 1s, 5s, 30s
CONF_QUANTILES = [0.01, 0.05, 0.10, 0.20, 0.50]
SIDES = ["short", "long", "both"]
MODES = ["market", "passive"]

# --- ES cost constants (CANONICAL) ---
ES_TICK_VALUE = 12.50
ES_TICK_POINTS = 0.25
ES_POINT_VALUE = 50.0  # $/pt
ES_RT_COMMISSION = 4.70
ES_RT_COMMISSION_TICKS = ES_RT_COMMISSION / ES_TICK_VALUE   # 0.376
ES_MARKET_COST_TICKS = ES_RT_COMMISSION_TICKS + 1.0          # 1.376
ES_PASSIVE_COST_TICKS = ES_RT_COMMISSION_TICKS               # 0.376

# --- MES cost constants ---
MES_TICK_VALUE = 1.25
MES_TICK_POINTS = 0.25
MES_POINT_VALUE = 5.0
MES_RT_COMMISSION = 1.50
MES_RT_COMMISSION_TICKS = MES_RT_COMMISSION / MES_TICK_VALUE  # 1.2
MES_MARKET_COST_TICKS = MES_RT_COMMISSION_TICKS + 1.0          # 2.2
MES_PASSIVE_COST_TICKS = MES_RT_COMMISSION_TICKS               # 1.2

PASSIVE_FILL_PROB = 0.50

# --- Regime / acceptance gates ---
REGIME_GAP_REJECT = 0.50
DAY_CONC_CAP = 0.70
REGIME_DAY_PCT_THRESHOLD = 0.10        # % close-to-close
MIN_TRADES = 30
MIN_SHARPE = 1.0
MIN_PF = 1.2

LOG_PATH = OUT_DIR / "run_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# =========================================================================
# Loaders
# =========================================================================

def load_es_pred_times(date_str: str) -> Optional[Dict]:
    """Load v3.4.2 predictions and map them to wall-clock timestamps via the
    ES MBO event index (WINDOW-1) + i*STRIDE."""
    es_path = ES_MBO_DIR / f"{date_str}_mbo_events.npz"
    pred_path = ES_PRED_DIR / f"oot_{date_str}.npz"
    if not es_path.exists() or not pred_path.exists():
        log.warning(f"missing data for {date_str}: mbo={es_path.exists()} pred={pred_path.exists()}")
        return None
    es = np.load(str(es_path), allow_pickle=True)
    timestamps = es["timestamps"]
    pf = np.load(str(pred_path), allow_pickle=True)
    n_pred = pf["pred_log_ret_1s"].shape[0]
    idx = (WINDOW - 1) + np.arange(n_pred) * STRIDE
    if idx[-1] >= len(timestamps):
        ok_n = int(np.searchsorted(idx, len(timestamps), side="left"))
        idx = idx[:ok_n]
        n_pred = ok_n
    pred_ts_ns = timestamps[idx]
    return {
        "pred_ts_ns": pred_ts_ns,
        "pred_log_ret_1s": pf["pred_log_ret_1s"][:n_pred],
        "pred_log_ret_5s": pf["pred_log_ret_5s"][:n_pred],
        "pred_log_ret_30s": pf["pred_log_ret_30s"][:n_pred],
    }


def load_es_trade_grid(date_str: str) -> Optional[Dict]:
    """Load ES last-trade ticks. Returns ts_ns and price_pts (float64 points)."""
    p = ES_TRADE_DIR / f"{date_str}_trades.npz"
    if not p.exists():
        log.warning(f"missing ES trade cache for {date_str}: {p}")
        return None
    d = np.load(str(p))
    return {
        "ts_ns": d["ts_ns"],
        "price_pts": d["price_raw"].astype(np.float64) / 1e9,
    }


def es_price_at(grid_ts: np.ndarray, price_pts: np.ndarray, query_ns: np.ndarray) -> np.ndarray:
    """Last trade price at-or-before each query ts. NaN before the first trade."""
    idx = np.searchsorted(grid_ts, query_ns, side="right") - 1
    valid = (idx >= 0) & (idx < len(grid_ts))
    out = np.full(len(query_ns), np.nan, dtype=np.float64)
    if valid.any():
        out[valid] = price_pts[idx[valid]]
    return out


# =========================================================================
# Per-day build: ES entry/exit prices for each horizon
# =========================================================================

def build_day_data(date_str: str) -> Optional[Dict]:
    es_pred = load_es_pred_times(date_str)
    es_px = load_es_trade_grid(date_str)
    if es_pred is None or es_px is None:
        return None
    pred_ts = es_pred["pred_ts_ns"]
    # Filter predictions whose entry+max_horizon fits inside the available
    # ES trade window.
    max_exit_offset_ns = max(HORIZONS_MS) * 1_000_000
    tmin = int(es_px["ts_ns"][0])
    tmax = int(es_px["ts_ns"][-1])
    in_win = (pred_ts >= tmin) & (pred_ts + max_exit_offset_ns <= tmax)
    n_in = int(in_win.sum())
    if n_in < 100:
        log.warning(f"  {date_str}: only {n_in} preds fit ES trade window — skipping")
        return None
    pred_ts = pred_ts[in_win]
    head_pred = {
        1000: es_pred["pred_log_ret_1s"][in_win],
        5000: es_pred["pred_log_ret_5s"][in_win],
        30000: es_pred["pred_log_ret_30s"][in_win],
    }
    # Entry price = ES at pred_ts. Exit = ES at pred_ts + h.
    entry_px = es_price_at(es_px["ts_ns"], es_px["price_pts"], pred_ts)
    exit_px_by_h = {}
    for h_ms in HORIZONS_MS:
        h_ns = int(h_ms) * 1_000_000
        ex = es_price_at(es_px["ts_ns"], es_px["price_pts"], pred_ts + h_ns)
        exit_px_by_h[h_ms] = ex
    # Regime: ES open vs close (first vs last trade in RTH window).
    pf = es_px["price_pts"]
    open_p = float(pf[0])
    close_p = float(pf[-1])
    day_pct = (close_p - open_p) / open_p * 100.0
    if day_pct > REGIME_DAY_PCT_THRESHOLD:
        regime = "green"
    elif day_pct < -REGIME_DAY_PCT_THRESHOLD:
        regime = "red"
    else:
        regime = "flat"
    return {
        "date": date_str,
        "regime": regime,
        "day_pct": day_pct,
        "n_preds": n_in,
        "pred_ts_ns": pred_ts,
        "head_pred": head_pred,
        "entry_px": entry_px,
        "exit_px_by_h": exit_px_by_h,
        "es_open": open_p,
        "es_close": close_p,
    }


# =========================================================================
# PnL per cell — in ES TICKS first, then convert to bps for each cost model.
# =========================================================================

def safe_sharpe(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    sd = float(x.std(ddof=1))
    if sd <= 0:
        return float("nan")
    return float(x.mean() / sd)


def safe_sortino(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    down = x[x < 0]
    if len(down) == 0:
        return float("inf") if x.mean() > 0 else 0.0
    dd = float(down.std(ddof=1)) if len(down) > 1 else float(abs(down[0]))
    if dd <= 0:
        return float("nan")
    return float(x.mean() / dd)


def safe_pf(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    pos = float(x[x > 0].sum())
    neg = float(-x[x < 0].sum())
    if neg <= 0:
        return float("inf") if pos > 0 else 0.0
    return pos / neg


def max_dd(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    eq = np.cumsum(x)
    peak = np.maximum.accumulate(eq)
    return float((eq - peak).min())


def evaluate_cell(days: List[Dict], h_ms: int, q: float, side: str, mode: str) -> Optional[Dict]:
    """Pool trades across days. Returns gross PnL in TICKS plus per-day rows.
    PnL conversion to bps for each cost model happens later (Part A vs Part B)."""
    pooled_gross_ticks = []
    pooled_entry_px = []
    per_day_rows = []
    for d in days:
        pred = d["head_pred"][h_ms]
        entry = d["entry_px"]
        exit_ = d["exit_px_by_h"][h_ms]
        ok = np.isfinite(pred) & np.isfinite(entry) & np.isfinite(exit_) & (entry > 0)
        if ok.sum() < 5:
            continue
        p_ok = pred[ok]
        e_ok = entry[ok]
        x_ok = exit_[ok]
        # Per-day per-side confidence threshold
        thr = np.percentile(np.abs(p_ok), 100.0 * (1.0 - q))
        sel = np.abs(p_ok) >= thr
        if side == "short":
            sel = sel & (p_ok < 0)
            sign = np.full(int(sel.sum()), -1.0)
        elif side == "long":
            sel = sel & (p_ok > 0)
            sign = np.full(int(sel.sum()), 1.0)
        else:
            sign = np.sign(p_ok[sel])
            sign[sign == 0] = 1.0
        if int(sel.sum()) < 3:
            continue
        # Raw move in ticks (1 tick = 0.25 pts)
        move_pts = x_ok[sel] - e_ok[sel]
        move_ticks = move_pts / ES_TICK_POINTS
        gross_ticks = move_ticks * sign  # signed PnL in ticks per contract
        entry_px_sel = e_ok[sel]
        pooled_gross_ticks.append(gross_ticks)
        pooled_entry_px.append(entry_px_sel)
        per_day_rows.append({
            "date": d["date"],
            "regime": d["regime"],
            "day_pct": d["day_pct"],
            "n_trades": int(sel.sum()),
            "gross_sum_ticks": float(gross_ticks.sum()),
            "gross_mean_ticks": float(gross_ticks.mean()),
            "entry_px_mean": float(entry_px_sel.mean()),
        })
    if not pooled_gross_ticks:
        return None
    gross_ticks = np.concatenate(pooled_gross_ticks)
    entry_px = np.concatenate(pooled_entry_px)
    n = len(gross_ticks)
    if n < 10:
        return None
    return {
        "horizon_ms": h_ms,
        "conf_q": q,
        "side": side,
        "mode": mode,
        "n_trades": n,
        "gross_ticks": gross_ticks,
        "entry_px": entry_px,
        "per_day_raw": per_day_rows,
    }


def cost_ticks_for(mode: str, contract: str) -> float:
    """Round-trip cost in ticks-of-that-contract."""
    if contract == "ES":
        return ES_MARKET_COST_TICKS if mode == "market" else ES_PASSIVE_COST_TICKS
    elif contract == "MES":
        return MES_MARKET_COST_TICKS if mode == "market" else MES_PASSIVE_COST_TICKS
    raise ValueError(contract)


def net_metrics_for(cell: Dict, contract: str) -> Dict:
    """Compute net Sharpe / Sortino / PF / WR / mean / dd / per-day rows in bps
    of the entry notional for the requested contract."""
    mode = cell["mode"]
    gross_ticks = cell["gross_ticks"].copy()
    entry_px = cell["entry_px"]
    cost_ticks_rt = cost_ticks_for(mode, contract)
    # Per-trade gross PnL in $: ticks * tick_value
    # Notional per trade in $: entry_px * point_value
    if contract == "ES":
        tick_val = ES_TICK_VALUE
        point_val = ES_POINT_VALUE
    else:
        tick_val = MES_TICK_VALUE
        point_val = MES_POINT_VALUE
    gross_usd = gross_ticks * tick_val
    notional_usd = entry_px * point_val
    cost_usd_rt = cost_ticks_rt * tick_val
    if mode == "passive":
        # Expectation modeling: only PASSIVE_FILL_PROB of trades fill. We scale
        # both gross and cost by the fill probability.
        gross_usd = gross_usd * PASSIVE_FILL_PROB
        cost_usd_rt = cost_usd_rt * PASSIVE_FILL_PROB
    net_usd = gross_usd - cost_usd_rt
    # bps of entry notional
    with np.errstate(divide="ignore", invalid="ignore"):
        gross_bps = np.where(notional_usd > 0, gross_usd / notional_usd * 1e4, np.nan)
        net_bps = np.where(notional_usd > 0, net_usd / notional_usd * 1e4, np.nan)
    # Aggregate per-day
    per_day = []
    cursor = 0
    for raw in cell["per_day_raw"]:
        n = raw["n_trades"]
        sl_net = net_bps[cursor:cursor + n]
        sl_gross = gross_bps[cursor:cursor + n]
        per_day.append({
            "date": raw["date"],
            "regime": raw["regime"],
            "day_pct": raw["day_pct"],
            "n_trades": n,
            "gross_mean_bps": float(np.nanmean(sl_gross)) if n else float("nan"),
            "net_sum_bps": float(np.nansum(sl_net)),
            "net_mean_bps": float(np.nanmean(sl_net)) if n else float("nan"),
            "net_sharpe": safe_sharpe(sl_net),
            "wr_net": float((sl_net > 0).mean()) if n else float("nan"),
        })
        cursor += n
    return {
        "contract": contract,
        "n_trades": len(net_bps),
        "gross_mean_bps": float(np.nanmean(gross_bps)),
        "net_mean_bps": float(np.nanmean(net_bps)),
        "net_median_bps": float(np.nanmedian(net_bps)),
        "net_p10_bps": float(np.nanpercentile(net_bps, 10)),
        "net_p90_bps": float(np.nanpercentile(net_bps, 90)),
        "net_std_bps": float(np.nanstd(net_bps, ddof=1)),
        "sharpe_net": safe_sharpe(net_bps),
        "sortino_net": safe_sortino(net_bps),
        "pf_net": safe_pf(net_bps),
        "wr_net": float(np.nanmean(net_bps > 0)),
        "max_dd_bps": max_dd(net_bps[np.isfinite(net_bps)]),
        "cost_ticks_rt": cost_ticks_rt,
        "cost_usd_rt": cost_ticks_rt * tick_val,
        "per_day": per_day,
    }


# =========================================================================
# Gate checks
# =========================================================================

def regime_gate(per_day: List[Dict]) -> Tuple[str, Dict]:
    by_regime = {"green": [], "red": [], "flat": []}
    for r in per_day:
        if r["regime"] in by_regime:
            by_regime[r["regime"]].append(r)
    info = {
        "n_green_days": len(by_regime["green"]),
        "n_red_days": len(by_regime["red"]),
        "n_flat_days": len(by_regime["flat"]),
    }

    def reg_shr(rs):
        if not rs:
            return float("nan")
        v = np.array([r["net_sharpe"] for r in rs if np.isfinite(r["net_sharpe"])])
        if len(v) == 0:
            return float("nan")
        return float(v.mean())
    sg = reg_shr(by_regime["green"])
    sr = reg_shr(by_regime["red"])
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
    abs_sums = np.array([abs(r["net_sum_bps"]) for r in per_day])
    total = abs_sums.sum()
    if total <= 0:
        return False, 0.0
    max_share = float(abs_sums.max() / total)
    return max_share <= DAY_CONC_CAP, max_share


def all_short_red_only_check(per_day: List[Dict], side: str) -> bool:
    if side != "short":
        return True
    regimes = set(r["regime"] for r in per_day if r["n_trades"] > 0)
    return bool(regimes - {"red"})


def survives_all(metrics: Dict, side: str) -> Tuple[bool, str]:
    if metrics["n_trades"] < MIN_TRADES:
        return False, f"FAIL_N({metrics['n_trades']})"
    if not (np.isfinite(metrics["sharpe_net"]) and metrics["sharpe_net"] > MIN_SHARPE):
        return False, f"FAIL_SHARPE({metrics['sharpe_net']:.3f})"
    if not (np.isfinite(metrics["pf_net"]) and metrics["pf_net"] > MIN_PF):
        return False, f"FAIL_PF({metrics['pf_net']:.3f})"
    rv, _ = regime_gate(metrics["per_day"])
    if rv != "PASS_REGIME":
        return False, rv
    ok_dc, _ = day_conc_check(metrics["per_day"])
    if not ok_dc:
        return False, "FAIL_DAYCONC"
    if not all_short_red_only_check(metrics["per_day"], side):
        return False, "FAIL_SHORT_RED_ONLY"
    return True, "SURV"


# =========================================================================
# Main
# =========================================================================

def main():
    t0 = time.time()
    log.info("=" * 78)
    log.info("es_mes_positive_control_v1 — Part A: ES native | Part B: MES re-cost")
    log.info(f"Dates: {DATES}")
    log.info(f"Grid: horizons={HORIZONS_MS}ms x conf_q={CONF_QUANTILES} x "
             f"sides={SIDES} x modes={MODES}")
    log.info(f"ES costs (RT ticks): market={ES_MARKET_COST_TICKS:.3f} passive={ES_PASSIVE_COST_TICKS:.3f}")
    log.info(f"MES costs (RT ticks): market={MES_MARKET_COST_TICKS:.3f} passive={MES_PASSIVE_COST_TICKS:.3f}")
    log.info("=" * 78)

    # --- Load days ---
    days = []
    for d_str in DATES:
        log.info(f"Loading {d_str}...")
        dd = build_day_data(d_str)
        if dd is None:
            log.warning(f"  {d_str}: SKIPPED")
            continue
        log.info(f"  {d_str}: regime={dd['regime']} day_pct={dd['day_pct']:+.3f}% "
                 f"n_preds={dd['n_preds']} es=[{dd['es_open']:.2f}->{dd['es_close']:.2f}]")
        days.append(dd)
    if not days:
        log.error("No valid days. Aborting.")
        return
    log.info(f"Loaded {len(days)} days; total preds = {sum(d['n_preds'] for d in days):,}")
    regime_counts = {r: 0 for r in ("green", "red", "flat")}
    for d in days:
        regime_counts[d["regime"]] += 1
    log.info(f"Regime composition: green={regime_counts['green']} "
             f"red={regime_counts['red']} flat={regime_counts['flat']}")

    # --- Evaluate every cell in the grid (gross is contract-agnostic) ---
    log.info("Building gross-PnL pools per cell...")
    cells = []
    total = len(HORIZONS_MS) * len(CONF_QUANTILES) * len(SIDES) * len(MODES)
    done = 0
    for h_ms in HORIZONS_MS:
        for q in CONF_QUANTILES:
            for side in SIDES:
                for mode in MODES:
                    base = evaluate_cell(days, h_ms, q, side, mode)
                    done += 1
                    if base is None:
                        continue
                    es_m = net_metrics_for(base, "ES")
                    mes_m = net_metrics_for(base, "MES")
                    es_surv, es_reason = survives_all(es_m, side)
                    mes_surv, mes_reason = survives_all(mes_m, side)
                    cells.append({
                        "horizon_ms": h_ms,
                        "conf_q": q,
                        "side": side,
                        "mode": mode,
                        "n_trades": base["n_trades"],
                        "n_days_active": len(base["per_day_raw"]),
                        "es": es_m,
                        "mes": mes_m,
                        "es_survives": es_surv,
                        "es_reason": es_reason,
                        "mes_survives": mes_surv,
                        "mes_reason": mes_reason,
                    })
    log.info(f"  evaluated {done}/{total} grid combinations; {len(cells)} produced enough trades")

    # --- Sort by ES Sharpe desc ---
    cells_sorted = sorted(
        cells,
        key=lambda c: (c["es"]["sharpe_net"] if np.isfinite(c["es"]["sharpe_net"]) else -1e9),
        reverse=True,
    )

    # --- Write results table ---
    fieldnames = [
        "horizon_ms", "conf_q", "side", "mode", "n_trades", "n_days_active",
        # ES net stats
        "es_sharpe_net", "es_sortino_net", "es_pf_net", "es_wr_net",
        "es_net_mean_bps", "es_net_median_bps", "es_max_dd_bps",
        "es_cost_ticks_rt", "es_cost_usd_rt",
        "es_survives", "es_reason",
        # MES net stats
        "mes_sharpe_net", "mes_sortino_net", "mes_pf_net", "mes_wr_net",
        "mes_net_mean_bps", "mes_net_median_bps", "mes_max_dd_bps",
        "mes_cost_ticks_rt", "mes_cost_usd_rt",
        "mes_survives", "mes_reason",
        # Gross
        "gross_mean_bps_es", "gross_mean_bps_mes",
    ]
    with open(OUT_DIR / "results_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for c in cells_sorted:
            row = {
                "horizon_ms": c["horizon_ms"], "conf_q": c["conf_q"], "side": c["side"],
                "mode": c["mode"], "n_trades": c["n_trades"], "n_days_active": c["n_days_active"],
                "es_sharpe_net": c["es"]["sharpe_net"], "es_sortino_net": c["es"]["sortino_net"],
                "es_pf_net": c["es"]["pf_net"], "es_wr_net": c["es"]["wr_net"],
                "es_net_mean_bps": c["es"]["net_mean_bps"], "es_net_median_bps": c["es"]["net_median_bps"],
                "es_max_dd_bps": c["es"]["max_dd_bps"],
                "es_cost_ticks_rt": c["es"]["cost_ticks_rt"], "es_cost_usd_rt": c["es"]["cost_usd_rt"],
                "es_survives": c["es_survives"], "es_reason": c["es_reason"],
                "mes_sharpe_net": c["mes"]["sharpe_net"], "mes_sortino_net": c["mes"]["sortino_net"],
                "mes_pf_net": c["mes"]["pf_net"], "mes_wr_net": c["mes"]["wr_net"],
                "mes_net_mean_bps": c["mes"]["net_mean_bps"], "mes_net_median_bps": c["mes"]["net_median_bps"],
                "mes_max_dd_bps": c["mes"]["max_dd_bps"],
                "mes_cost_ticks_rt": c["mes"]["cost_ticks_rt"], "mes_cost_usd_rt": c["mes"]["cost_usd_rt"],
                "mes_survives": c["mes_survives"], "mes_reason": c["mes_reason"],
                "gross_mean_bps_es": c["es"]["gross_mean_bps"], "gross_mean_bps_mes": c["mes"]["gross_mean_bps"],
            }
            w.writerow(row)
    log.info(f"Wrote results_table.csv ({len(cells_sorted)} rows)")

    # --- Per-day breakdown for top-50 ES cells ---
    perday_rows = []
    for c in cells_sorted[:50]:
        for r in c["es"]["per_day"]:
            perday_rows.append({
                "horizon_ms": c["horizon_ms"], "conf_q": c["conf_q"],
                "side": c["side"], "mode": c["mode"], "contract": "ES",
                "date": r["date"], "regime": r["regime"], "day_pct": r["day_pct"],
                "n_trades": r["n_trades"],
                "gross_mean_bps": r["gross_mean_bps"], "net_sum_bps": r["net_sum_bps"],
                "net_mean_bps": r["net_mean_bps"], "net_sharpe": r["net_sharpe"],
                "wr_net": r["wr_net"],
            })
        for r in c["mes"]["per_day"]:
            perday_rows.append({
                "horizon_ms": c["horizon_ms"], "conf_q": c["conf_q"],
                "side": c["side"], "mode": c["mode"], "contract": "MES",
                "date": r["date"], "regime": r["regime"], "day_pct": r["day_pct"],
                "n_trades": r["n_trades"],
                "gross_mean_bps": r["gross_mean_bps"], "net_sum_bps": r["net_sum_bps"],
                "net_mean_bps": r["net_mean_bps"], "net_sharpe": r["net_sharpe"],
                "wr_net": r["wr_net"],
            })
    if perday_rows:
        with open(OUT_DIR / "per_day_breakdown.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(perday_rows[0].keys()))
            w.writeheader()
            w.writerows(perday_rows)
        log.info(f"Wrote per_day_breakdown.csv ({len(perday_rows)} rows, top-50 ES cells x both contracts)")

    # --- Survivors tally ---
    es_survivors = [c for c in cells_sorted if c["es_survives"]]
    mes_survivors = [c for c in cells_sorted if c["mes_survives"]]
    both_survivors = [c for c in cells_sorted if c["es_survives"] and c["mes_survives"]]
    log.info("=" * 78)
    log.info(f"ES survivors (all gates): {len(es_survivors)}")
    log.info(f"MES survivors (all gates): {len(mes_survivors)}")
    log.info(f"Survivors at BOTH ES and MES cost: {len(both_survivors)}")
    log.info("=" * 78)

    # --- Top 5 ES (Part A) ---
    log.info("PART A — TOP 5 by ES net Sharpe:")
    hdr = (f"  {'cell':<48}{'n':<7}{'ES_Shr':<9}{'ES_PF':<8}{'ES_WR':<8}"
           f"{'ES_bps':<10}{'gate'}")
    log.info(hdr)
    top5_lines = []
    for c in cells_sorted[:5]:
        cell_id = (f"h{c['horizon_ms']/1000:g}s_q{int(c['conf_q']*100)}_"
                   f"{c['side']}_{c['mode']}")
        gate = "SURV" if c["es_survives"] else c["es_reason"]
        line = (f"  {cell_id:<48}{c['n_trades']:<7d}"
                f"{c['es']['sharpe_net']:<9.3f}{c['es']['pf_net']:<8.2f}"
                f"{c['es']['wr_net']:<8.2%}{c['es']['net_mean_bps']:<10.3f}{gate}")
        log.info(line)
        top5_lines.append(line)

    # --- MES degradation on Part-A surviving cells (Part B) ---
    log.info("PART B — MES re-cost of ES survivors:")
    if es_survivors:
        hdrb = (f"  {'cell':<48}{'ES_Shr':<9}{'MES_Shr':<10}"
                f"{'ES_bps':<10}{'MES_bps':<10}{'MES_gate'}")
        log.info(hdrb)
        b_lines = []
        for c in es_survivors:
            cell_id = (f"h{c['horizon_ms']/1000:g}s_q{int(c['conf_q']*100)}_"
                       f"{c['side']}_{c['mode']}")
            mes_gate = "SURV" if c["mes_survives"] else c["mes_reason"]
            line = (f"  {cell_id:<48}{c['es']['sharpe_net']:<9.3f}"
                    f"{c['mes']['sharpe_net']:<10.3f}"
                    f"{c['es']['net_mean_bps']:<10.3f}"
                    f"{c['mes']['net_mean_bps']:<10.3f}{mes_gate}")
            log.info(line)
            b_lines.append(line)
    else:
        log.info("  (no ES survivors to re-cost)")
        b_lines = []

    # --- Write survivors.txt ---
    with open(OUT_DIR / "survivors.txt", "w") as f:
        f.write("=" * 78 + "\n")
        f.write("es_mes_positive_control_v1 — survivors summary\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Dates: {DATES}\n")
        f.write(f"Days loaded: {len(days)} (green={regime_counts['green']} "
                f"red={regime_counts['red']} flat={regime_counts['flat']})\n\n")
        f.write(f"ES survivors: {len(es_survivors)}\n")
        f.write(f"MES survivors: {len(mes_survivors)}\n")
        f.write(f"Both: {len(both_survivors)}\n\n")
        f.write("PART A — Top 5 ES cells by Sharpe:\n")
        f.write("\n".join(top5_lines) + "\n\n")
        f.write("PART B — MES re-cost of ES survivors:\n")
        f.write("\n".join(b_lines) + "\n\n")
        f.write("Full ES survivors:\n")
        for c in es_survivors:
            cell_id = (f"h{c['horizon_ms']/1000:g}s_q{int(c['conf_q']*100)}_"
                       f"{c['side']}_{c['mode']}")
            f.write(f"  {cell_id}  ES_Shr={c['es']['sharpe_net']:.3f} "
                    f"ES_PF={c['es']['pf_net']:.2f} ES_bps={c['es']['net_mean_bps']:.3f} "
                    f"n={c['n_trades']} | MES_Shr={c['mes']['sharpe_net']:.3f} "
                    f"MES_PF={c['mes']['pf_net']:.2f} MES_bps={c['mes']['net_mean_bps']:.3f} "
                    f"mes_gate={'SURV' if c['mes_survives'] else c['mes_reason']}\n")

    # --- VERDICT ---
    log.info("=" * 78)
    if es_survivors:
        log.info("VERDICT — PART A: POSITIVE CONTROL CONFIRMED")
        log.info(f"  {len(es_survivors)} ES cell(s) pass all gates on the 6-day window.")
        log.info("  -> Signal v3.4.2 is still profitable executed natively on ES.")
        log.info("  -> Cross-asset SPY failure was purely signal-transfer loss, NOT signal decay.")
    else:
        log.info("VERDICT — PART A: POSITIVE CONTROL FAILED")
        log.info("  No ES cell passes all gates. Top 5 listed above for diagnostic.")
        log.info("  -> Signal v3.4.2 cannot clear ES execution costs on this window.")
        log.info("  -> Implication: signal decayed OR window is regime-adverse.")
    if mes_survivors:
        log.info(f"VERDICT — PART B: MES VIABLE ({len(mes_survivors)} cells survive at MES cost)")
    else:
        log.info("VERDICT — PART B: MES NOT VIABLE on this window.")
    log.info("=" * 78)

    # --- MLflow ---
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("es_mes_positive_control_v1")
        with mlflow.start_run(run_name=f"run_{time.strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("dates", ",".join(DATES))
            mlflow.log_param("n_days", len(days))
            mlflow.log_param("horizons_ms", HORIZONS_MS)
            mlflow.log_param("conf_quantiles", CONF_QUANTILES)
            mlflow.log_param("sides", SIDES)
            mlflow.log_param("modes", MODES)
            mlflow.log_param("passive_fill_prob", PASSIVE_FILL_PROB)
            mlflow.log_param("es_market_cost_ticks", ES_MARKET_COST_TICKS)
            mlflow.log_param("es_passive_cost_ticks", ES_PASSIVE_COST_TICKS)
            mlflow.log_param("mes_market_cost_ticks", MES_MARKET_COST_TICKS)
            mlflow.log_param("mes_passive_cost_ticks", MES_PASSIVE_COST_TICKS)
            mlflow.log_metric("total_cells", len(cells_sorted))
            mlflow.log_metric("es_survivors", len(es_survivors))
            mlflow.log_metric("mes_survivors", len(mes_survivors))
            mlflow.log_metric("both_survivors", len(both_survivors))
            if cells_sorted:
                top = cells_sorted[0]
                mlflow.log_metric("best_es_sharpe", top["es"]["sharpe_net"])
                mlflow.log_metric("best_es_pf", top["es"]["pf_net"])
                mlflow.log_metric("best_es_net_mean_bps", top["es"]["net_mean_bps"])
                mlflow.log_metric("best_mes_sharpe", top["mes"]["sharpe_net"])
                mlflow.log_metric("best_mes_pf", top["mes"]["pf_net"])
                mlflow.log_metric("best_mes_net_mean_bps", top["mes"]["net_mean_bps"])
            mlflow.log_artifact(str(OUT_DIR / "results_table.csv"))
            mlflow.log_artifact(str(OUT_DIR / "survivors.txt"))
            mlflow.log_artifact(str(OUT_DIR / "run_log.txt"))
            if (OUT_DIR / "per_day_breakdown.csv").exists():
                mlflow.log_artifact(str(OUT_DIR / "per_day_breakdown.csv"))
        log.info("MLflow logging complete.")
    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")

    log.info(f"DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
