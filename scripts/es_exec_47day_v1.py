#!/usr/bin/env python3
"""es_exec_47day_v1.py — Full-OOT ES/MES execution grid with per-day Sharpe gate.

CONTEXT
=======
Follow-up to es_mes_positive_control_v1, which ran on only 6 days and used a
per-trade Sharpe > 1.0 gate. Two methodology fixes:

  1. HC #428 R1 — OOT validation must use the FULL OOT window (>=40 days),
     not a 6-day SPY-overlap subset. The actual cnn_mamba_v3_4_2_fixedmtl
     OOT directory contains 34 per-date npz files (Feb 23 - Apr 14 2026).
     One (20260308) is a Sunday with no trade cache, leaving 33 usable days.

  2. HC #69 — per-trade Sharpe is the wrong gate for HFT. The standard
     industry metric is per-day Sharpe (annualized, mean_dpnl / std_dpnl
     * sqrt(252)). A per-trade Sharpe of 0.1 with 100 trades/day compounds
     to per-day Sharpe ~= 1.0. This script makes per-day Sharpe the
     PRIMARY metric and uses 1.5 as the real-world HFT bar.

GRID (HC #428 R2 head-matched)
==============================
  horizon       in {1s, 5s, 30s}
  conf_quantile in {top1%, top5%, top10%, top20%, top50%}
  side          in {short, long, both}
  exec_mode     in {ES_market, ES_passive, MES_market, MES_passive}

CANONICAL COSTS (CLAUDE.md)
===========================
  ES tick=$12.50  RT comm=$4.70=0.376 ticks
    market RT  = 1.376 ticks  ~ 0.75 bps on $230k notional
    passive RT = 0.376 ticks  ~ 0.20 bps (50% fill -> halve gross AND cost)
  MES tick=$1.25  RT comm=$1.50=1.2 ticks
    market RT  = 2.20 ticks   ~ 1.20 bps on $23k notional
    passive RT = 1.20 ticks   ~ 0.65 bps (50% fill -> halve gross AND cost)

GATES (HC #428 R1 + R2 + HC #344 + HC #69)
==========================================
  - per-day Sharpe (annualized) > 1.5   (PRIMARY)
  - per-day PF > 1.4
  - per-day WR > 0.55
  - regime gap: |Sharpe_green - Sharpe_red| / max <= 0.50
  - day concentration: max |day_pnl| / sum|day_pnl| <= 0.70
  - n_days_traded >= 30
  - not all-short-on-red-only

OUTPUTS
=======
  output/es_exec_47day_v1/results_table.csv
  output/es_exec_47day_v1/per_day_breakdown.csv
  output/es_exec_47day_v1/survivors.txt
  output/es_exec_47day_v1/run_log.txt
MLflow experiment: es_exec_47day_v1
"""
from __future__ import annotations

import csv
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
ES_PRED_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
ES_MBO_DIR = ROOT / "data/processed/mbo_events_smart_v3"
ES_TRADE_DIR = ROOT / "data/derived/mid_price_cache_hc439"
OUT_DIR = ROOT / "output/es_exec_47day_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 1000
STRIDE = 250

HORIZONS_MS = [1000, 5000, 30000]
CONF_QUANTILES = [0.01, 0.05, 0.10, 0.20, 0.50]
SIDES = ["short", "long", "both"]
EXEC_MODES = ["ES_market", "ES_passive", "MES_market", "MES_passive"]

# Cost constants
ES_TICK_VALUE = 12.50
ES_TICK_POINTS = 0.25
ES_POINT_VALUE = 50.0
ES_RT_COMMISSION = 4.70
ES_RT_COMMISSION_TICKS = ES_RT_COMMISSION / ES_TICK_VALUE
ES_MARKET_COST_TICKS = ES_RT_COMMISSION_TICKS + 1.0
ES_PASSIVE_COST_TICKS = ES_RT_COMMISSION_TICKS

MES_TICK_VALUE = 1.25
MES_TICK_POINTS = 0.25
MES_POINT_VALUE = 5.0
MES_RT_COMMISSION = 1.50
MES_RT_COMMISSION_TICKS = MES_RT_COMMISSION / MES_TICK_VALUE
MES_MARKET_COST_TICKS = MES_RT_COMMISSION_TICKS + 1.0
MES_PASSIVE_COST_TICKS = MES_RT_COMMISSION_TICKS

PASSIVE_FILL_PROB = 0.50

# Gates
MIN_PER_DAY_SHARPE = 1.5
MIN_PER_DAY_PF = 1.4
MIN_PER_DAY_WR = 0.55
REGIME_GAP_REJECT = 0.50
DAY_CONC_CAP = 0.70
REGIME_DAY_PCT_THRESHOLD = 0.10
MIN_DAYS_TRADED = 30

TRADING_DAYS_PER_YEAR = 252

LOG_PATH = OUT_DIR / "run_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# =========================================================================
# Discover usable dates: pred + MBO + trade cache all present
# =========================================================================

def discover_dates() -> List[str]:
    preds = sorted(
        f.name.replace("oot_", "").replace(".npz", "")
        for f in ES_PRED_DIR.iterdir() if f.is_file() and f.name.startswith("oot_") and f.name.endswith(".npz")
    )
    keep = []
    for d in preds:
        if not (ES_MBO_DIR / f"{d}_mbo_events.npz").exists():
            log.warning(f"  drop {d}: no MBO events")
            continue
        if not (ES_TRADE_DIR / f"{d}_trades.npz").exists():
            log.warning(f"  drop {d}: no trade cache")
            continue
        keep.append(d)
    return keep


# =========================================================================
# Loaders
# =========================================================================

def load_es_pred_times(date_str: str) -> Optional[Dict]:
    es_path = ES_MBO_DIR / f"{date_str}_mbo_events.npz"
    pred_path = ES_PRED_DIR / f"oot_{date_str}.npz"
    try:
        es = np.load(str(es_path), allow_pickle=True)
        timestamps = es["timestamps"]
        pf = np.load(str(pred_path), allow_pickle=True)
        keys = set(pf.files)
    except Exception as e:
        log.warning(f"  {date_str}: load failure {e}")
        return None
    needed = ["pred_log_ret_1s", "pred_log_ret_5s", "pred_log_ret_30s"]
    for k in needed:
        if k not in keys:
            log.warning(f"  {date_str}: pred missing key {k}; keys={sorted(keys)}")
            return None
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
    p = ES_TRADE_DIR / f"{date_str}_trades.npz"
    try:
        d = np.load(str(p))
    except Exception as e:
        log.warning(f"  {date_str}: trade-cache load failure {e}")
        return None
    if "ts_ns" not in d.files or "price_raw" not in d.files:
        log.warning(f"  {date_str}: trade cache missing fields; have={sorted(d.files)}")
        return None
    return {
        "ts_ns": d["ts_ns"],
        "price_pts": d["price_raw"].astype(np.float64) / 1e9,
    }


def es_price_at(grid_ts: np.ndarray, price_pts: np.ndarray, query_ns: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(grid_ts, query_ns, side="right") - 1
    valid = (idx >= 0) & (idx < len(grid_ts))
    out = np.full(len(query_ns), np.nan, dtype=np.float64)
    if valid.any():
        out[valid] = price_pts[idx[valid]]
    return out


# =========================================================================
# Per-day build
# =========================================================================

def build_day_data(date_str: str) -> Optional[Dict]:
    pred = load_es_pred_times(date_str)
    px = load_es_trade_grid(date_str)
    if pred is None or px is None:
        return None
    pred_ts = pred["pred_ts_ns"]
    max_exit_offset_ns = max(HORIZONS_MS) * 1_000_000
    if len(px["ts_ns"]) == 0:
        log.warning(f"  {date_str}: empty trade grid")
        return None
    tmin = int(px["ts_ns"][0])
    tmax = int(px["ts_ns"][-1])
    in_win = (pred_ts >= tmin) & (pred_ts + max_exit_offset_ns <= tmax)
    n_in = int(in_win.sum())
    if n_in < 100:
        log.warning(f"  {date_str}: only {n_in} preds in window — skipping")
        return None
    pred_ts = pred_ts[in_win]
    head_pred = {
        1000: pred["pred_log_ret_1s"][in_win],
        5000: pred["pred_log_ret_5s"][in_win],
        30000: pred["pred_log_ret_30s"][in_win],
    }
    entry_px = es_price_at(px["ts_ns"], px["price_pts"], pred_ts)
    exit_px_by_h = {}
    for h_ms in HORIZONS_MS:
        h_ns = int(h_ms) * 1_000_000
        exit_px_by_h[h_ms] = es_price_at(px["ts_ns"], px["price_pts"], pred_ts + h_ns)
    open_p = float(px["price_pts"][0])
    close_p = float(px["price_pts"][-1])
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
# Cell evaluation: pool per-trade ticks across days, keeping per-day slices.
# =========================================================================

def cost_ticks_for_mode(exec_mode: str) -> Tuple[float, str]:
    if exec_mode == "ES_market":
        return ES_MARKET_COST_TICKS, "ES"
    if exec_mode == "ES_passive":
        return ES_PASSIVE_COST_TICKS, "ES"
    if exec_mode == "MES_market":
        return MES_MARKET_COST_TICKS, "MES"
    if exec_mode == "MES_passive":
        return MES_PASSIVE_COST_TICKS, "MES"
    raise ValueError(exec_mode)


def contract_constants(contract: str) -> Tuple[float, float]:
    if contract == "ES":
        return ES_TICK_VALUE, ES_POINT_VALUE
    if contract == "MES":
        return MES_TICK_VALUE, MES_POINT_VALUE
    raise ValueError(contract)


def evaluate_cell(days: List[Dict], h_ms: int, q: float, side: str, exec_mode: str) -> Optional[Dict]:
    """Return per-day rows and aggregate metrics for this grid cell."""
    cost_ticks_rt, contract = cost_ticks_for_mode(exec_mode)
    tick_val, point_val = contract_constants(contract)
    is_passive = "passive" in exec_mode
    cost_usd_rt = cost_ticks_rt * tick_val
    cost_usd_rt_effective = cost_usd_rt * (PASSIVE_FILL_PROB if is_passive else 1.0)
    fill_scale = PASSIVE_FILL_PROB if is_passive else 1.0

    per_day = []
    all_trade_bps = []
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
        thr = np.percentile(np.abs(p_ok), 100.0 * (1.0 - q))
        sel = np.abs(p_ok) >= thr
        if side == "short":
            sel = sel & (p_ok < 0)
            sign = np.full(int(sel.sum()), -1.0)
        elif side == "long":
            sel = sel & (p_ok > 0)
            sign = np.full(int(sel.sum()), 1.0)
        else:
            s = np.sign(p_ok[sel])
            s[s == 0] = 1.0
            sign = s
        n_sel = int(sel.sum())
        if n_sel < 3:
            continue
        move_pts = x_ok[sel] - e_ok[sel]
        move_ticks = move_pts / ES_TICK_POINTS  # ES point grid; both ES and MES use same 0.25 grid
        gross_ticks_signed = move_ticks * sign
        # USD per trade in this contract
        gross_usd = gross_ticks_signed * tick_val * fill_scale
        net_usd = gross_usd - cost_usd_rt_effective
        entry_px_sel = e_ok[sel]
        notional_usd = entry_px_sel * point_val
        with np.errstate(divide="ignore", invalid="ignore"):
            net_bps = np.where(notional_usd > 0, net_usd / notional_usd * 1e4, np.nan)
            gross_bps = np.where(notional_usd > 0, gross_usd / notional_usd * 1e4, np.nan)
        day_net_sum_usd = float(np.nansum(net_usd))
        day_net_mean_bps = float(np.nanmean(net_bps)) if n_sel else float("nan")
        day_gross_mean_bps = float(np.nanmean(gross_bps)) if n_sel else float("nan")
        # Per-trade Sharpe on this day's bps (diagnostic)
        finite = net_bps[np.isfinite(net_bps)]
        if len(finite) >= 2 and finite.std(ddof=1) > 0:
            day_per_trade_sharpe = float(finite.mean() / finite.std(ddof=1))
        else:
            day_per_trade_sharpe = float("nan")
        day_wr = float((net_bps > 0).mean()) if n_sel else float("nan")
        per_day.append({
            "date": d["date"],
            "regime": d["regime"],
            "day_pct": d["day_pct"],
            "n_trades": n_sel,
            "gross_mean_bps": day_gross_mean_bps,
            "net_sum_bps": float(np.nansum(net_bps)),
            "net_mean_bps": day_net_mean_bps,
            "net_sum_usd": day_net_sum_usd,
            "per_trade_sharpe": day_per_trade_sharpe,
            "wr_net": day_wr,
            "entry_px_mean": float(np.nanmean(entry_px_sel)),
        })
        all_trade_bps.append(net_bps)

    if not per_day:
        return None
    n_days_traded = len(per_day)
    if n_days_traded < 2:
        return None

    # ----- PER-DAY metric aggregation (PRIMARY) -----
    daily_pnl_usd = np.array([r["net_sum_usd"] for r in per_day])
    daily_net_mean_bps = np.array([r["net_mean_bps"] for r in per_day])
    daily_net_sum_bps = np.array([r["net_sum_bps"] for r in per_day])

    mean_dpnl = float(daily_pnl_usd.mean())
    std_dpnl = float(daily_pnl_usd.std(ddof=1)) if n_days_traded > 1 else 0.0
    if std_dpnl > 0:
        per_day_sharpe = mean_dpnl / std_dpnl * math.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        per_day_sharpe = float("nan")
    downside = daily_pnl_usd[daily_pnl_usd < 0]
    if len(downside) >= 1:
        # downside std with ddof=0 if single point, else ddof=1
        d_std = float(downside.std(ddof=1)) if len(downside) > 1 else float(abs(downside[0]))
        per_day_sortino = (mean_dpnl / d_std * math.sqrt(TRADING_DAYS_PER_YEAR)) if d_std > 0 else float("nan")
    else:
        per_day_sortino = float("inf") if mean_dpnl > 0 else 0.0
    pos = float(daily_pnl_usd[daily_pnl_usd > 0].sum())
    neg = float(-daily_pnl_usd[daily_pnl_usd < 0].sum())
    per_day_pf = (pos / neg) if neg > 0 else (float("inf") if pos > 0 else 0.0)
    per_day_wr = float((daily_pnl_usd > 0).mean())

    # Day-concentration based on $ PnL
    abs_d = np.abs(daily_pnl_usd)
    day_conc = float(abs_d.max() / abs_d.sum()) if abs_d.sum() > 0 else 1.0

    # Regime split per-day Sharpes
    by_reg = {"green": [], "red": [], "flat": []}
    for row in per_day:
        by_reg[row["regime"]].append(row["net_sum_usd"])

    def reg_shr(arr):
        a = np.array(arr)
        if len(a) < 2:
            return float("nan")
        sd = a.std(ddof=1)
        if sd <= 0:
            return float("nan")
        return float(a.mean() / sd * math.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe_green = reg_shr(by_reg["green"])
    sharpe_red = reg_shr(by_reg["red"])
    sharpe_flat = reg_shr(by_reg["flat"])

    # Drawdown ($, on cumulative daily PnL)
    eq = np.cumsum(daily_pnl_usd)
    peak = np.maximum.accumulate(eq)
    max_dd_usd = float((eq - peak).min()) if len(eq) else 0.0
    # bps drawdown on per-trade flow
    all_bps = np.concatenate(all_trade_bps)
    finite_bps = all_bps[np.isfinite(all_bps)]
    if len(finite_bps):
        eq_bps = np.cumsum(finite_bps)
        peak_bps = np.maximum.accumulate(eq_bps)
        max_dd_bps = float((eq_bps - peak_bps).min())
    else:
        max_dd_bps = float("nan")

    n_trades = int(sum(r["n_trades"] for r in per_day))
    avg_tpd = n_trades / n_days_traded

    # Per-trade Sharpe across all trades (diagnostic / secondary)
    if len(finite_bps) >= 2 and finite_bps.std(ddof=1) > 0:
        per_trade_sharpe = float(finite_bps.mean() / finite_bps.std(ddof=1))
    else:
        per_trade_sharpe = float("nan")
    per_trade_net_mean_bps = float(np.nanmean(all_bps))

    return {
        "horizon_ms": h_ms,
        "conf_q": q,
        "side": side,
        "exec_mode": exec_mode,
        "contract": contract,
        "cost_ticks_rt": cost_ticks_rt,
        "cost_usd_rt": cost_usd_rt,
        "fill_scale": fill_scale,
        # counts
        "n_trades": n_trades,
        "n_days_traded": n_days_traded,
        "avg_trades_per_day": avg_tpd,
        # PRIMARY
        "per_day_sharpe": per_day_sharpe,
        "per_day_sortino": per_day_sortino,
        "per_day_pf": per_day_pf,
        "per_day_wr": per_day_wr,
        "mean_daily_pnl_usd": mean_dpnl,
        "std_daily_pnl_usd": std_dpnl,
        "mean_daily_net_bps": float(daily_net_sum_bps.mean()),
        # SECONDARY (diagnostic)
        "per_trade_sharpe": per_trade_sharpe,
        "per_trade_net_mean_bps": per_trade_net_mean_bps,
        # regime
        "sharpe_green": sharpe_green,
        "sharpe_red": sharpe_red,
        "sharpe_flat": sharpe_flat,
        "n_green_days": len(by_reg["green"]),
        "n_red_days": len(by_reg["red"]),
        "n_flat_days": len(by_reg["flat"]),
        # risk
        "max_dd_usd": max_dd_usd,
        "max_dd_bps": max_dd_bps,
        "day_conc": day_conc,
        "per_day": per_day,
    }


# =========================================================================
# Gates
# =========================================================================

def evaluate_gates(c: Dict) -> Tuple[bool, List[str]]:
    fails = []
    if c["n_days_traded"] < MIN_DAYS_TRADED:
        fails.append(f"FAIL_DAYS({c['n_days_traded']})")
    s = c["per_day_sharpe"]
    if not (np.isfinite(s) and s > MIN_PER_DAY_SHARPE):
        fails.append(f"FAIL_SHARPE({s:.2f})")
    pf = c["per_day_pf"]
    if not (np.isfinite(pf) and pf > MIN_PER_DAY_PF):
        fails.append(f"FAIL_PF({pf:.2f})")
    wr = c["per_day_wr"]
    if not (np.isfinite(wr) and wr > MIN_PER_DAY_WR):
        fails.append(f"FAIL_WR({wr:.2%})")
    sg, sr = c["sharpe_green"], c["sharpe_red"]
    if not (np.isfinite(sg) and np.isfinite(sr)):
        fails.append("FAIL_REGIME_COVERAGE")
    else:
        denom = max(abs(sg), abs(sr))
        if denom == 0:
            fails.append("FAIL_REGIME_FLAT")
        else:
            gap = abs(sg - sr) / denom
            if gap > REGIME_GAP_REJECT:
                fails.append(f"FAIL_REGIME_GAP({gap:.2f})")
    if c["day_conc"] > DAY_CONC_CAP:
        fails.append(f"FAIL_DAYCONC({c['day_conc']:.2f})")
    if c["side"] == "short":
        regimes_traded = set(r["regime"] for r in c["per_day"] if r["n_trades"] > 0)
        if regimes_traded == {"red"}:
            fails.append("FAIL_SHORT_RED_ONLY")
    return (len(fails) == 0), fails


# =========================================================================
# Main
# =========================================================================

def main():
    t0 = time.time()
    log.info("=" * 78)
    log.info("es_exec_47day_v1 — FULL OOT ES/MES execution grid (per-day Sharpe gate)")
    log.info(f"Grid: H={HORIZONS_MS}ms x q={CONF_QUANTILES} x sides={SIDES} x exec={EXEC_MODES}")
    log.info(f"ES costs (RT ticks):  market={ES_MARKET_COST_TICKS:.3f}  passive={ES_PASSIVE_COST_TICKS:.3f}")
    log.info(f"MES costs (RT ticks): market={MES_MARKET_COST_TICKS:.3f} passive={MES_PASSIVE_COST_TICKS:.3f}")
    log.info(f"Gates: per_day_Sharpe>{MIN_PER_DAY_SHARPE}, PF>{MIN_PER_DAY_PF}, WR>{MIN_PER_DAY_WR:.0%}, "
             f"regime_gap<={REGIME_GAP_REJECT}, day_conc<={DAY_CONC_CAP}, n_days>={MIN_DAYS_TRADED}")
    log.info("=" * 78)

    dates = discover_dates()
    log.info(f"Discovered {len(dates)} fully-covered OOT dates")
    log.info(f"  first={dates[0]} last={dates[-1]}")

    days = []
    skipped = []
    for d_str in dates:
        dd = build_day_data(d_str)
        if dd is None:
            skipped.append(d_str)
            continue
        log.info(f"  {d_str}: regime={dd['regime']:<5} day_pct={dd['day_pct']:+.3f}% "
                 f"n_preds={dd['n_preds']:>5} es=[{dd['es_open']:.2f}->{dd['es_close']:.2f}]")
        days.append(dd)
    if skipped:
        log.warning(f"Skipped {len(skipped)} dates: {skipped}")
    if not days:
        log.error("No valid days. Aborting.")
        return
    regime_counts = {"green": 0, "red": 0, "flat": 0}
    for d in days:
        regime_counts[d["regime"]] += 1
    log.info(f"Loaded {len(days)} days; regime: green={regime_counts['green']} "
             f"red={regime_counts['red']} flat={regime_counts['flat']}")
    log.info(f"Total predictions: {sum(d['n_preds'] for d in days):,}")

    # -------- Sweep --------
    log.info("Sweeping grid...")
    cells = []
    total = len(HORIZONS_MS) * len(CONF_QUANTILES) * len(SIDES) * len(EXEC_MODES)
    done = 0
    for h_ms in HORIZONS_MS:
        for q in CONF_QUANTILES:
            for side in SIDES:
                for exec_mode in EXEC_MODES:
                    c = evaluate_cell(days, h_ms, q, side, exec_mode)
                    done += 1
                    if c is None:
                        continue
                    surv, fails = evaluate_gates(c)
                    c["survives"] = surv
                    c["fail_reasons"] = ";".join(fails) if fails else "SURV"
                    cells.append(c)
    log.info(f"  evaluated {done}/{total} grid combinations; {len(cells)} produced enough trades")

    # -------- Write results table --------
    fields = [
        "horizon_ms", "horizon_s", "conf_q", "side", "exec_mode", "contract",
        "n_trades", "n_days_traded", "avg_trades_per_day",
        "per_day_sharpe", "per_day_sortino", "per_day_pf", "per_day_wr",
        "mean_daily_pnl_usd", "std_daily_pnl_usd", "mean_daily_net_bps",
        "per_trade_sharpe", "per_trade_net_mean_bps",
        "sharpe_green", "sharpe_red", "sharpe_flat",
        "n_green_days", "n_red_days", "n_flat_days",
        "max_dd_usd", "max_dd_bps", "day_conc",
        "cost_ticks_rt", "cost_usd_rt",
        "survives", "fail_reasons",
    ]
    cells_sorted = sorted(
        cells,
        key=lambda c: (c["per_day_sharpe"] if np.isfinite(c["per_day_sharpe"]) else -1e9),
        reverse=True,
    )
    with open(OUT_DIR / "results_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in cells_sorted:
            w.writerow({
                "horizon_ms": c["horizon_ms"],
                "horizon_s": c["horizon_ms"] / 1000.0,
                "conf_q": c["conf_q"],
                "side": c["side"],
                "exec_mode": c["exec_mode"],
                "contract": c["contract"],
                "n_trades": c["n_trades"],
                "n_days_traded": c["n_days_traded"],
                "avg_trades_per_day": c["avg_trades_per_day"],
                "per_day_sharpe": c["per_day_sharpe"],
                "per_day_sortino": c["per_day_sortino"],
                "per_day_pf": c["per_day_pf"],
                "per_day_wr": c["per_day_wr"],
                "mean_daily_pnl_usd": c["mean_daily_pnl_usd"],
                "std_daily_pnl_usd": c["std_daily_pnl_usd"],
                "mean_daily_net_bps": c["mean_daily_net_bps"],
                "per_trade_sharpe": c["per_trade_sharpe"],
                "per_trade_net_mean_bps": c["per_trade_net_mean_bps"],
                "sharpe_green": c["sharpe_green"],
                "sharpe_red": c["sharpe_red"],
                "sharpe_flat": c["sharpe_flat"],
                "n_green_days": c["n_green_days"],
                "n_red_days": c["n_red_days"],
                "n_flat_days": c["n_flat_days"],
                "max_dd_usd": c["max_dd_usd"],
                "max_dd_bps": c["max_dd_bps"],
                "day_conc": c["day_conc"],
                "cost_ticks_rt": c["cost_ticks_rt"],
                "cost_usd_rt": c["cost_usd_rt"],
                "survives": c["survives"],
                "fail_reasons": c["fail_reasons"],
            })
    log.info(f"Wrote results_table.csv ({len(cells_sorted)} rows)")

    # -------- Per-day breakdown for top-30 + survivors --------
    survivors = [c for c in cells_sorted if c["survives"]]
    perday_targets = list(cells_sorted[:30])
    seen = {(c["horizon_ms"], c["conf_q"], c["side"], c["exec_mode"]) for c in perday_targets}
    for c in survivors:
        k = (c["horizon_ms"], c["conf_q"], c["side"], c["exec_mode"])
        if k not in seen:
            perday_targets.append(c)
            seen.add(k)
    perday_rows = []
    for c in perday_targets:
        for r in c["per_day"]:
            perday_rows.append({
                "horizon_ms": c["horizon_ms"],
                "conf_q": c["conf_q"],
                "side": c["side"],
                "exec_mode": c["exec_mode"],
                "contract": c["contract"],
                "date": r["date"],
                "regime": r["regime"],
                "day_pct": r["day_pct"],
                "n_trades": r["n_trades"],
                "gross_mean_bps": r["gross_mean_bps"],
                "net_sum_bps": r["net_sum_bps"],
                "net_mean_bps": r["net_mean_bps"],
                "net_sum_usd": r["net_sum_usd"],
                "per_trade_sharpe": r["per_trade_sharpe"],
                "wr_net": r["wr_net"],
                "survives_overall": c["survives"],
            })
    if perday_rows:
        with open(OUT_DIR / "per_day_breakdown.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(perday_rows[0].keys()))
            w.writeheader()
            w.writerows(perday_rows)
        log.info(f"Wrote per_day_breakdown.csv ({len(perday_rows)} rows)")

    # -------- Top 5 by per-day Sharpe --------
    log.info("=" * 78)
    log.info("TOP 5 by per-day Sharpe (annualized):")
    hdr = (f"  {'cell':<48}{'days':<6}{'n':<8}{'pdShr':<8}{'PF':<7}{'WR':<8}"
           f"{'$/day':<10}{'gate'}")
    log.info(hdr)
    top5_pdshr_lines = []
    for c in cells_sorted[:5]:
        cell_id = (f"h{c['horizon_ms']/1000:g}s_q{int(c['conf_q']*100)}_"
                   f"{c['side']}_{c['exec_mode']}")
        line = (f"  {cell_id:<48}{c['n_days_traded']:<6d}{c['n_trades']:<8d}"
                f"{c['per_day_sharpe']:<8.2f}{c['per_day_pf']:<7.2f}"
                f"{c['per_day_wr']:<8.2%}{c['mean_daily_pnl_usd']:<10.1f}"
                f"{c['fail_reasons']}")
        log.info(line)
        top5_pdshr_lines.append(line)

    # -------- Top 5 by mean daily net bps --------
    cells_by_bps = sorted(
        cells_sorted,
        key=lambda c: (c["mean_daily_net_bps"] if np.isfinite(c["mean_daily_net_bps"]) else -1e9),
        reverse=True,
    )
    log.info("TOP 5 by mean daily net bps:")
    log.info(hdr)
    top5_bps_lines = []
    for c in cells_by_bps[:5]:
        cell_id = (f"h{c['horizon_ms']/1000:g}s_q{int(c['conf_q']*100)}_"
                   f"{c['side']}_{c['exec_mode']}")
        line = (f"  {cell_id:<48}{c['n_days_traded']:<6d}{c['n_trades']:<8d}"
                f"{c['per_day_sharpe']:<8.2f}{c['per_day_pf']:<7.2f}"
                f"{c['per_day_wr']:<8.2%}{c['mean_daily_pnl_usd']:<10.1f}"
                f"{c['fail_reasons']}")
        log.info(line)
        top5_bps_lines.append(line)

    log.info("=" * 78)
    log.info(f"SURVIVORS (all gates): {len(survivors)}")
    for c in survivors:
        cell_id = (f"h{c['horizon_ms']/1000:g}s_q{int(c['conf_q']*100)}_"
                   f"{c['side']}_{c['exec_mode']}")
        log.info(f"  {cell_id}  pdShr={c['per_day_sharpe']:.2f} PF={c['per_day_pf']:.2f} "
                 f"WR={c['per_day_wr']:.2%} $/day={c['mean_daily_pnl_usd']:.2f} "
                 f"trades/day={c['avg_trades_per_day']:.1f}")
    log.info("=" * 78)

    # -------- survivors.txt --------
    with open(OUT_DIR / "survivors.txt", "w") as f:
        f.write("=" * 78 + "\n")
        f.write("es_exec_47day_v1 — survivors summary\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"OOT dates: {len(days)} days "
                f"({dates[0]} -> {dates[-1]})\n")
        f.write(f"Regime composition: green={regime_counts['green']} "
                f"red={regime_counts['red']} flat={regime_counts['flat']}\n\n")
        f.write(f"GATES (HC #428):\n")
        f.write(f"  per_day_Sharpe > {MIN_PER_DAY_SHARPE}\n")
        f.write(f"  per_day_PF     > {MIN_PER_DAY_PF}\n")
        f.write(f"  per_day_WR     > {MIN_PER_DAY_WR}\n")
        f.write(f"  regime_gap    <= {REGIME_GAP_REJECT}\n")
        f.write(f"  day_conc      <= {DAY_CONC_CAP}\n")
        f.write(f"  n_days        >= {MIN_DAYS_TRADED}\n\n")
        f.write(f"SURVIVORS: {len(survivors)}\n\n")
        f.write("TOP 5 by per-day Sharpe:\n")
        f.write("\n".join(top5_pdshr_lines) + "\n\n")
        f.write("TOP 5 by mean daily net bps:\n")
        f.write("\n".join(top5_bps_lines) + "\n\n")
        if survivors:
            f.write("FULL SURVIVOR LIST:\n")
            for c in survivors:
                cell_id = (f"h{c['horizon_ms']/1000:g}s_q{int(c['conf_q']*100)}_"
                           f"{c['side']}_{c['exec_mode']}")
                f.write(
                    f"  {cell_id}\n"
                    f"    per_day_Sharpe={c['per_day_sharpe']:.2f}  Sortino={c['per_day_sortino']:.2f}  "
                    f"PF={c['per_day_pf']:.2f}  WR={c['per_day_wr']:.2%}\n"
                    f"    mean_$/day={c['mean_daily_pnl_usd']:.2f}  std_$/day={c['std_daily_pnl_usd']:.2f}  "
                    f"max_DD=${c['max_dd_usd']:.2f}\n"
                    f"    n_trades={c['n_trades']}  n_days={c['n_days_traded']}  "
                    f"trades/day={c['avg_trades_per_day']:.1f}\n"
                    f"    regime: green_Shr={c['sharpe_green']:.2f}  red_Shr={c['sharpe_red']:.2f}  "
                    f"flat_Shr={c['sharpe_flat']:.2f}\n"
                    f"    day_conc={c['day_conc']:.2%}  cost_ticks_RT={c['cost_ticks_rt']:.3f}\n\n"
                )
        else:
            f.write("(no cells pass all gates)\n")

    # -------- MLflow --------
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("es_exec_47day_v1")
        with mlflow.start_run(run_name=f"run_{time.strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("n_dates", len(days))
            mlflow.log_param("date_first", dates[0])
            mlflow.log_param("date_last", dates[-1])
            mlflow.log_param("horizons_ms", HORIZONS_MS)
            mlflow.log_param("conf_quantiles", CONF_QUANTILES)
            mlflow.log_param("sides", SIDES)
            mlflow.log_param("exec_modes", EXEC_MODES)
            mlflow.log_param("min_per_day_sharpe", MIN_PER_DAY_SHARPE)
            mlflow.log_param("min_per_day_pf", MIN_PER_DAY_PF)
            mlflow.log_param("min_per_day_wr", MIN_PER_DAY_WR)
            mlflow.log_metric("n_cells_evaluated", len(cells_sorted))
            mlflow.log_metric("n_survivors", len(survivors))
            if cells_sorted:
                top = cells_sorted[0]
                mlflow.log_metric("best_per_day_sharpe", top["per_day_sharpe"])
                mlflow.log_metric("best_per_day_pf", top["per_day_pf"])
                mlflow.log_metric("best_mean_daily_pnl_usd", top["mean_daily_pnl_usd"])
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
