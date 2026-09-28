#!/usr/bin/env python3
"""ES-SPY basis mean-reversion pair-trade research (v1).

Idea: ES futures vs SPY ETF are highly correlated. The "basis" residual
basis_t = ES_t - k*SPY_t should mean-revert. Trade when |z| > threshold.

Inputs:
  - ES trade prints: /data/derived/mid_price_cache_hc439/<DATE>_trades.npz
    (last trade price at each 250ms boundary used as ES mid proxy)
  - SPY mid grid: /data/processed/spy_mid_grid/<DATE>_mid_250ms.npz

Cost model (RT both legs combined):
  - ES (full): $29.70 RT / ~$225k notional = ~1.3 bps
  - SPY: ~$0.02 spread + 0.28 bps SEC ~= 4.7 bps RT
  - Total RT ~ 6 bps. Use 6 bps as base, sensitivity to 12 bps.

Outputs to /output/es_spy_basis_pair_v1/.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/es_spy_basis_pair_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPY_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/spy_mid_grid")
ES_TRADE_DIR = Path("/home/jupiter/Lvl3Quant/data/derived/mid_price_cache_hc439")

# Overlap RTH days (ES MBO Mar 1-9 plus 10-12 trade caches; SPY grid Mar 2-12)
# Actual ES MBO only Mar 1-6,8,9. Drop Mar 1 (no SPY), Mar 8 (no SPY/Sunday).
# Trade caches exist for Mar 10-12 -- include if BOTH files present.
CANDIDATE_DATES = [
    "20260302", "20260303", "20260304", "20260305", "20260306",
    "20260309", "20260310", "20260311", "20260312",
]

GRID_MS = 250
GRID_NS = GRID_MS * 1_000_000

# Pair-trade grid
Z_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0]
HOLD_SECONDS = [5, 30, 60, 300]
SIDES = ["both", "long_basis", "short_basis"]  # long_basis = long ES short SPY (basis cheap)

# Z-score rolling window
Z_WINDOW_SEC = 5 * 60  # 5 min
Z_WINDOW_BARS = (Z_WINDOW_SEC * 1000) // GRID_MS  # 1200 bars at 250ms

# Cost (round-trip total for both legs, in bps of notional traded)
COST_BPS_RT_BASE = 6.0  # base
COST_BPS_RT_HIGH = 12.0  # sensitivity

# Exit when z reverts to this fraction of entry z (sign-aware)
EXIT_Z_FRAC = 0.25


# ---------------------------------------------------------------------------
def load_es_grid(date_str: str, rth_start: int, rth_end: int) -> np.ndarray:
    """Build ES 250ms grid via 'last trade price <= boundary' reduction.

    Returns array of length n_grid with ES price in points (NaN if no trade
    yet at that boundary).
    """
    cache = ES_TRADE_DIR / f"{date_str}_trades.npz"
    if not cache.exists():
        raise FileNotFoundError(f"ES trade cache missing: {cache}")
    d = np.load(cache)
    ts_ns = d["ts_ns"]
    px = d["price_raw"].astype(np.float64) / 1e9

    n_grid = int((rth_end - rth_start) // GRID_NS)
    boundaries = rth_start + np.arange(n_grid, dtype=np.int64) * GRID_NS

    # searchsorted: index of first trade > boundary, so last <= boundary is idx-1
    idx = np.searchsorted(ts_ns, boundaries, side="right") - 1
    out = np.full(n_grid, np.nan, dtype=np.float64)
    valid = idx >= 0
    out[valid] = px[idx[valid]]
    return out, boundaries


def load_spy_grid(date_str: str):
    p = SPY_DIR / f"{date_str}_mid_250ms.npz"
    if not p.exists():
        raise FileNotFoundError(f"SPY grid missing: {p}")
    d = np.load(p)
    stats = json.loads(str(d["stats"][0]))
    return (d["grid_ts_ns"], d["mid_price"].astype(np.float64),
            d["bid_price"].astype(np.float64), d["ask_price"].astype(np.float64),
            stats)


def rolling_zscore(x: np.ndarray, w: int) -> np.ndarray:
    """Rolling z-score with window w (right-aligned). First w-1 entries NaN."""
    n = len(x)
    z = np.full(n, np.nan, dtype=np.float64)
    if n < w:
        return z
    # Cumulative sums for O(n) rolling mean / std
    cs = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x * x)])
    win_sum = cs[w:] - cs[:-w]
    win_sum2 = cs2[w:] - cs2[:-w]
    mean = win_sum / w
    var = win_sum2 / w - mean * mean
    var = np.clip(var, 1e-12, None)
    std = np.sqrt(var)
    z[w - 1:] = (x[w - 1:] - mean) / std
    return z


# ---------------------------------------------------------------------------
@dataclass
class DayResult:
    date: str
    n_trades: int
    gross_bps: float
    net_bps: float
    wins: int
    losses: int
    pnl_per_trade_bps: list
    classification: str  # green / red / flat
    es_close_to_close_pct: float


def classify_day(es_grid: np.ndarray) -> tuple[str, float]:
    """ES close-to-close % move -> green/red/flat."""
    valid = ~np.isnan(es_grid)
    if not valid.any():
        return "flat", 0.0
    first = es_grid[valid][0]
    last = es_grid[valid][-1]
    pct = (last - first) / first * 100.0
    if pct > 0.10:
        return "green", pct
    if pct < -0.10:
        return "red", pct
    return "flat", pct


def simulate_day(es: np.ndarray,
                 spy: np.ndarray,
                 z_thresh: float,
                 hold_sec: int,
                 side: str,
                 cost_bps_rt: float) -> DayResult:
    """Simulate basis pair-trade on one day. Returns DayResult.

    Trade definition (one round-trip):
      - At step t: if z[t] > +z_thresh and side allows 'short_basis': enter short basis
        (short ES leg, long SPY leg).
      - If z[t] < -z_thresh and side allows 'long_basis': enter long basis.
      - Exit when |z| < EXIT_Z_FRAC * z_thresh OR after hold_sec.
      - PnL = change in basis as % of ES_entry, scaled by direction:
          long_basis_pnl_bps = (basis_exit - basis_entry) / es_entry * 1e4
          short_basis_pnl_bps = -(basis_exit - basis_entry) / es_entry * 1e4
      - Subtract cost_bps_rt once per trade (round-trip both legs).

    We assume capital symmetric across legs (ES leg notional = SPY leg notional).
    The basis change is measured in price units; we convert to bps relative
    to ES_t notional (since both legs are equal notional, the basis change is
    the alpha extracted on the gross both-leg notional we'll express in bps
    of single-leg notional ~ ES_t).
    """
    n = len(es)
    valid = ~(np.isnan(es) | np.isnan(spy))
    if valid.sum() < Z_WINDOW_BARS + 10:
        return DayResult(date="", n_trades=0, gross_bps=0.0, net_bps=0.0,
                         wins=0, losses=0, pnl_per_trade_bps=[],
                         classification="flat", es_close_to_close_pct=0.0)

    # Fit k via OLS on first 30 min of day (avoid look-ahead)
    fit_bars = (30 * 60 * 1000) // GRID_MS  # 7200 bars
    fit_end = min(fit_bars, valid.sum())
    es_v = es[valid][:fit_end]
    spy_v = spy[valid][:fit_end]
    # k = cov / var (no intercept; basis residual will include intercept)
    spy_dm = spy_v - spy_v.mean()
    es_dm = es_v - es_v.mean()
    var_spy = (spy_dm * spy_dm).sum()
    if var_spy < 1e-9:
        return DayResult(date="", n_trades=0, gross_bps=0.0, net_bps=0.0,
                         wins=0, losses=0, pnl_per_trade_bps=[],
                         classification="flat", es_close_to_close_pct=0.0)
    k = (spy_dm * es_dm).sum() / var_spy

    # Basis on full series (forward-only k from morning fit)
    basis = es - k * spy
    # Z-score
    # Need to be careful with NaN; for simplicity, treat NaN basis as carry-forward
    # before z-score, since trade-print last-known-good is reasonable.
    basis_f = basis.copy()
    if np.isnan(basis_f[0]):
        # find first valid
        first_v = np.argmax(~np.isnan(basis_f))
        basis_f[:first_v] = basis_f[first_v] if not np.isnan(basis_f[first_v]) else 0.0
    # Forward-fill NaNs
    for i in range(1, n):
        if np.isnan(basis_f[i]):
            basis_f[i] = basis_f[i - 1]

    z = rolling_zscore(basis_f, Z_WINDOW_BARS)

    # Hold in bars
    hold_bars = (hold_sec * 1000) // GRID_MS

    # Simulate -- no overlapping trades. Walk forward.
    pnl_list = []
    t = Z_WINDOW_BARS
    while t < n - hold_bars - 1:
        z_t = z[t]
        if np.isnan(z_t):
            t += 1
            continue

        direction = 0
        if z_t > z_thresh and side in ("both", "short_basis"):
            direction = -1  # short basis (basis high -> will fall)
        elif z_t < -z_thresh and side in ("both", "long_basis"):
            direction = +1  # long basis (basis low -> will rise)

        if direction == 0:
            t += 1
            continue

        # Entry: must have valid es/spy at t
        if np.isnan(es[t]) or np.isnan(spy[t]):
            t += 1
            continue

        basis_entry = basis_f[t]
        es_entry = es[t]
        # Find exit
        exit_idx = t + hold_bars
        target_z_abs = EXIT_Z_FRAC * z_thresh
        for j in range(t + 1, min(t + hold_bars + 1, n)):
            if not np.isnan(z[j]) and abs(z[j]) < target_z_abs:
                exit_idx = j
                break
            # Sign cross (z crossed zero): definitely exit
            if not np.isnan(z[j]) and np.sign(z[j]) != np.sign(z_t):
                exit_idx = j
                break

        exit_idx = min(exit_idx, n - 1)
        basis_exit = basis_f[exit_idx]
        # PnL: direction * (basis_exit - basis_entry) for long_basis means
        # basis going up = profit. But careful with sign:
        # long_basis = bet basis rises (we entered when z < 0, basis below mean)
        # -> direction=+1, profit when basis_exit > basis_entry. OK.
        # short_basis = bet basis falls (entered when z > 0)
        # -> direction=-1, profit when basis_exit < basis_entry. OK.
        delta = direction * (basis_exit - basis_entry)
        # Convert price-units delta to bps of ES notional (single-leg).
        # Since both legs are equal notional, the alpha vs combined 2x notional
        # would be delta / (2 * es_entry). To be conservative we measure vs
        # single-leg notional (es_entry).
        gross_bps = (delta / es_entry) * 1e4
        net_bps = gross_bps - cost_bps_rt
        pnl_list.append(net_bps)
        # Skip past exit + small buffer to avoid overlap
        t = exit_idx + 4  # 1s buffer
    arr = np.array(pnl_list, dtype=np.float64) if pnl_list else np.array([])
    gross = float(arr.sum() + cost_bps_rt * len(arr)) if len(arr) else 0.0
    net = float(arr.sum()) if len(arr) else 0.0
    wins = int((arr > 0).sum()) if len(arr) else 0
    losses = int((arr <= 0).sum()) if len(arr) else 0
    cls, pct = classify_day(es)
    return DayResult(
        date="",
        n_trades=len(arr),
        gross_bps=gross,
        net_bps=net,
        wins=wins,
        losses=losses,
        pnl_per_trade_bps=arr.tolist(),
        classification=cls,
        es_close_to_close_pct=pct,
    )


# ---------------------------------------------------------------------------
def compute_metrics(per_trade: list[float]) -> dict:
    """Sharpe / Sortino / PF / WR from per-trade bps PnL (treating each trade
    as one observation). Sharpe annualization done on a per-trade basis is
    awkward; here we report per-trade Sharpe (mean/std) and gross/net cumulative.
    """
    a = np.array(per_trade, dtype=np.float64)
    n = len(a)
    if n == 0:
        return {"n": 0, "sharpe": 0.0, "sortino": 0.0, "pf": 0.0, "wr": 0.0,
                "mean_bps": 0.0, "sum_bps": 0.0, "std_bps": 0.0}
    mean = a.mean()
    std = a.std(ddof=1) if n > 1 else 1.0
    downside = a[a < 0]
    dstd = downside.std(ddof=1) if len(downside) > 1 else (abs(downside.mean()) if len(downside) else 1.0)
    sharpe = mean / std if std > 1e-9 else 0.0
    sortino = mean / dstd if dstd > 1e-9 else 0.0
    gains = a[a > 0].sum()
    losses_abs = -a[a < 0].sum()
    pf = gains / losses_abs if losses_abs > 1e-9 else (gains if gains > 0 else 0.0)
    wr = float((a > 0).sum() / n)
    return {"n": n, "sharpe": float(sharpe), "sortino": float(sortino),
            "pf": float(pf), "wr": wr,
            "mean_bps": float(mean), "sum_bps": float(a.sum()),
            "std_bps": float(std)}


def per_regime_metrics(day_results: list[DayResult]) -> dict:
    """Sharpe per regime classification."""
    regimes = {}
    for cls in ("green", "red", "flat"):
        trades = []
        for dr in day_results:
            if dr.classification == cls:
                trades.extend(dr.pnl_per_trade_bps)
        regimes[cls] = compute_metrics(trades)
    return regimes


# ---------------------------------------------------------------------------
def discover_dates() -> list[str]:
    dates = []
    for d in CANDIDATE_DATES:
        es_ok = (ES_TRADE_DIR / f"{d}_trades.npz").exists()
        spy_ok = (SPY_DIR / f"{d}_mid_250ms.npz").exists()
        if es_ok and spy_ok:
            dates.append(d)
    return dates


def run_grid(dates: list[str], cost_bps_rt: float) -> list[dict]:
    """Run full z x hold x side grid, return list of cell dicts."""
    # Pre-load all data
    day_data = {}
    for d in dates:
        spy_ts, spy_mid, spy_bid, spy_ask, stats = load_spy_grid(d)
        rth_start = int(stats["rth_start_ns"])
        rth_end = int(stats["rth_end_ns"])
        es_grid, boundaries = load_es_grid(d, rth_start, rth_end)
        # Align lengths
        n = min(len(es_grid), len(spy_mid))
        day_data[d] = {
            "es": es_grid[:n],
            "spy": spy_mid[:n],
            "rth_start": rth_start,
        }
        v_es = (~np.isnan(es_grid[:n])).sum()
        v_spy = (~np.isnan(spy_mid[:n])).sum()
        print(f"  {d}: n_grid={n} es_valid={v_es} spy_valid={v_spy}", flush=True)

    cells = []
    for z in Z_THRESHOLDS:
        for h in HOLD_SECONDS:
            for s in SIDES:
                day_results = []
                for d in dates:
                    res = simulate_day(day_data[d]["es"], day_data[d]["spy"],
                                       z, h, s, cost_bps_rt)
                    res.date = d
                    day_results.append(res)

                all_trades = []
                for dr in day_results:
                    all_trades.extend(dr.pnl_per_trade_bps)
                overall = compute_metrics(all_trades)
                regimes = per_regime_metrics(day_results)
                per_day = [
                    {"date": dr.date, "n": dr.n_trades,
                     "net_bps": dr.net_bps, "gross_bps": dr.gross_bps,
                     "cls": dr.classification,
                     "es_pct": dr.es_close_to_close_pct}
                    for dr in day_results
                ]

                # Regime gate (HC #428 R1)
                sg = regimes.get("green", {}).get("sharpe", 0.0)
                sr = regimes.get("red", {}).get("sharpe", 0.0)
                denom = max(abs(sg), abs(sr), 1e-9)
                regime_disp = abs(sg - sr) / denom

                cell = {
                    "z_thresh": z,
                    "hold_sec": h,
                    "side": s,
                    "cost_bps_rt": cost_bps_rt,
                    "overall": overall,
                    "regimes": regimes,
                    "regime_disparity": float(regime_disp),
                    "per_day": per_day,
                    "regime_gate_pass": bool(regime_disp <= 0.5
                                             and overall["n"] >= 10),
                }
                cells.append(cell)

    return cells


def rank_cells(cells: list[dict]) -> list[dict]:
    """Sort by overall Sharpe (only cells with N>=10 trades)."""
    eligible = [c for c in cells if c["overall"]["n"] >= 10]
    eligible.sort(key=lambda c: c["overall"]["sharpe"], reverse=True)
    return eligible


def maybe_log_mlflow(cells: list[dict], dates: list[str], top: list[dict],
                     cost_bps_rt: float) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("es_spy_basis_pair_v1")
        with mlflow.start_run(run_name=f"basis_pair_cost{cost_bps_rt}bps"):
            mlflow.log_param("dates", ",".join(dates))
            mlflow.log_param("n_dates", len(dates))
            mlflow.log_param("z_window_sec", Z_WINDOW_SEC)
            mlflow.log_param("grid_ms", GRID_MS)
            mlflow.log_param("cost_bps_rt", cost_bps_rt)
            mlflow.log_param("exit_z_frac", EXIT_Z_FRAC)
            mlflow.log_param("z_thresholds", str(Z_THRESHOLDS))
            mlflow.log_param("hold_seconds", str(HOLD_SECONDS))
            mlflow.log_param("sides", str(SIDES))
            mlflow.log_param("n_cells", len(cells))
            mlflow.log_param("n_eligible", len(top))
            if top:
                best = top[0]
                mlflow.log_metric("best_sharpe", best["overall"]["sharpe"])
                mlflow.log_metric("best_sortino", best["overall"]["sortino"])
                mlflow.log_metric("best_pf", best["overall"]["pf"])
                mlflow.log_metric("best_wr", best["overall"]["wr"])
                mlflow.log_metric("best_net_bps", best["overall"]["sum_bps"])
                mlflow.log_metric("best_n_trades", best["overall"]["n"])
                mlflow.log_metric("best_regime_disparity", best["regime_disparity"])
            # Log results JSON as artifact
            art = OUT_DIR / f"all_cells_cost{cost_bps_rt}bps.json"
            mlflow.log_artifact(str(art))
    except Exception as e:
        print(f"[mlflow] skipped: {e}", flush=True)


# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    dates = discover_dates()
    print(f"Dates with both ES+SPY: {dates}", flush=True)
    if len(dates) < 3:
        print("ERROR: too few overlap days", file=sys.stderr)
        sys.exit(1)

    summary = {"dates": dates, "runs": []}
    for cost in (COST_BPS_RT_BASE, COST_BPS_RT_HIGH):
        print(f"\n=== Running cost={cost} bps RT ===", flush=True)
        cells = run_grid(dates, cost)
        top = rank_cells(cells)
        # Save all cells
        all_path = OUT_DIR / f"all_cells_cost{cost}bps.json"
        with open(all_path, "w") as f:
            json.dump(cells, f, indent=1, default=str)
        # Save top 5
        top_path = OUT_DIR / f"top5_cost{cost}bps.json"
        with open(top_path, "w") as f:
            json.dump(top[:5], f, indent=2, default=str)

        # Print top 5
        print(f"\nTop 5 cells at cost={cost} bps RT:")
        print(f"{'rank':<5}{'z':<6}{'hold':<8}{'side':<14}{'N':<7}{'Sharpe':<9}{'Sortino':<9}{'PF':<7}{'WR':<7}{'sumBps':<10}{'regDisp':<8}{'gatePass'}")
        for i, c in enumerate(top[:5]):
            o = c["overall"]
            print(f"{i+1:<5}{c['z_thresh']:<6}{c['hold_sec']:<8}{c['side']:<14}"
                  f"{o['n']:<7}{o['sharpe']:<9.3f}{o['sortino']:<9.3f}{o['pf']:<7.2f}"
                  f"{o['wr']:<7.2%}{o['sum_bps']:<10.1f}{c['regime_disparity']:<8.2f}"
                  f"{c['regime_gate_pass']}")

        summary["runs"].append({
            "cost_bps_rt": cost,
            "n_cells": len(cells),
            "n_eligible": len(top),
            "top5": [{"z": c["z_thresh"], "hold": c["hold_sec"],
                       "side": c["side"], **c["overall"],
                       "regime_disparity": c["regime_disparity"],
                       "gate_pass": c["regime_gate_pass"]}
                      for c in top[:5]],
        })
        maybe_log_mlflow(cells, dates, top, cost)

    summary["elapsed_sec"] = time.time() - t_start
    sum_path = OUT_DIR / "summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {sum_path}", flush=True)
    print(f"Done in {summary['elapsed_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
