"""
wheel_v8_robustness_v1.py — Parameter-sensitivity + tail-stress robustness of
the CANONICAL v8_WF wheel config (Tier2_Balanced_FW, walk-forward skew,
slippage on, regime overlay, real-blend IV, 2020-2025).

PRE-REGISTERED DESIGN (written before any cell is run — 2026-06-11):

Question: is the canonical Tier2_Balanced_FW (realized Sharpe 1.49 /
CAGR 13.8% / MaxDD -8.4%) a parameter PLATEAU (robust) or a KNIFE EDGE
(curve-fit)? Prior sensitivity work (tier_sensitivity_v1) only perturbed
IV level and slippage — never the strategy parameters themselves.

Cells (one-at-a-time perturbations of the v8_WF baseline, ~±20% per knob):
    baseline      put/call Δ 0.22, DTE 30-45, PT 0.65, roll 1, names 18,
                  ivr floor 0.25, slip ×1     (exact v8_WF replication)
    delta_018     put/call Δ 0.18              (-18%)
    delta_026     put/call Δ 0.26              (+18%)
    dte_24_36     DTE 24-36                    (-20%)
    dte_36_54     DTE 36-54                    (+20%)
    pt_052        profit_take 0.52             (-20%)
    pt_078        profit_take 0.78             (+20%)
    roll_3        roll_dte_trigger 3           (structural: base 1 can't -20%)
    ivr_020       iv_rank_floor 0.20           (-20%)
    ivr_030       iv_rank_floor 0.30           (+20%)
    names_14      max_concurrent_names 14      (-22%)
    names_22      max_concurrent_names 22      (+22%)
    slip_x2       slippage ×2 (5% of premium/leg, $0.06 min)   (cost stress)
    slip_x3       slippage ×3                                   (cost stress)
    slip_x4       slippage ×4                                   (cost stress)

Per cell we report (REALIZED-CASH primary, MTM secondary):
    Sharpe / Sortino / CAGR / MaxDD / Calmar / PF / WR / trades,
    HC #428 R1 regime stratification (SPY green/red/flat day Sharpe + gap),
    per-year realized Sharpe/return/MaxDD,
    crisis windows: COVID 2020-02-19..2020-04-30, 2022 bear
    2022-01-03..2022-10-14, Aug-2024 unwind 2024-07-15..2024-08-15,
    day concentration (best day positive pnl / total positive pnl, HC #344
    cap 0.70) and ticker concentration (best ticker pnl share).

PRE-REGISTERED VERDICT RULES:
    - Baseline replication must land within ±0.05 realized Sharpe of the
      canonical 1.49, else the harness is wrong and everything is void.
    - A knob is FRAGILE if either ±20% perturbation gives
      realized Sharpe < 0.50 × baseline OR realized MaxDD worse than
      2× baseline MaxDD.
    - Config verdict ROBUST if no knob is fragile AND slippage breakeven
      multiple (linear interp of realized CAGR -> 0) >= 2.0× assumed cost.
      Otherwise FRAGILE-<list of knobs>.

SLIDING-WINDOW note (HC #0): this is not a walk-forward fit — no parameter
is being SELECTED here. The skew calibration inside is the leak-free
walk-forward schedule (v8_WF). All evaluation windows are fixed calendar
strata of the same 2020-2025 simulation.

Usage (from /home/jupiter/Lvl3Quant/wheel_strategy_v1):
    python3 -m strategy.wheel_v8_robustness_v1 \
        --out results/wheel_v8_robustness_v1 --workers 3
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace, asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

START = "2020-01-01"
END = "2025-12-31"
CAPITAL = 100_000.0

CRISIS_WINDOWS = {
    "covid_2020": ("2020-02-19", "2020-04-30"),
    "bear_2022": ("2022-01-03", "2022-10-14"),
    "unwind_aug2024": ("2024-07-15", "2024-08-15"),
}

# ---------------- cell grid (pre-registered) ----------------
# (cell_name, cfg_overrides, iv_rank_floor_override, slip_mult)
CELLS = [
    ("baseline",  {},                                   None, 1.0),
    ("delta_018", {"put_delta_target": 0.18, "call_delta_target": 0.18}, None, 1.0),
    ("delta_026", {"put_delta_target": 0.26, "call_delta_target": 0.26}, None, 1.0),
    ("dte_24_36", {"dte_min": 24, "dte_max": 36},       None, 1.0),
    ("dte_36_54", {"dte_min": 36, "dte_max": 54},       None, 1.0),
    ("pt_052",    {"profit_take_pct": 0.52},            None, 1.0),
    ("pt_078",    {"profit_take_pct": 0.78},            None, 1.0),
    ("roll_3",    {"roll_dte_trigger": 3},              None, 1.0),
    ("ivr_020",   {},                                   0.20, 1.0),
    ("ivr_030",   {},                                   0.30, 1.0),
    ("names_14",  {"max_concurrent_names": 14},         None, 1.0),
    ("names_22",  {"max_concurrent_names": 22},         None, 1.0),
    ("slip_x2",   {},                                   None, 2.0),
    ("slip_x3",   {},                                   None, 3.0),
    ("slip_x4",   {},                                   None, 4.0),
]

KNOB_OF_CELL = {
    "delta_018": "delta", "delta_026": "delta",
    "dte_24_36": "dte", "dte_36_54": "dte",
    "pt_052": "profit_take", "pt_078": "profit_take",
    "roll_3": "roll_trigger",
    "ivr_020": "iv_rank_floor", "ivr_030": "iv_rank_floor",
    "names_14": "max_names", "names_22": "max_names",
}

# ---------------- per-process worker state ----------------
_W = {}


def _worker_init():
    """Load data + v8_WF engine state ONCE per worker process."""
    from strategy import iv_skew as _ivs
    from strategy.tier_runner import _load_inputs
    from strategy.regime_overlay import build_regime, apply_regime_gate

    _ivs.load_calibration_walkforward()  # leak-free WF skew schedule (v8_WF)
    data = _load_inputs(modeled=False, smoke=False, real_iv=True)
    regime = build_regime()
    data["macro"] = apply_regime_gate(data["macro"], regime,
                                      vix_force_gate=999.0)
    _W["data"] = data


def _realized_daily(led: pd.DataFrame, dates: pd.DatetimeIndex,
                    capital: float) -> pd.Series:
    """Realized-cash equity series indexed by date (same as tier_runner)."""
    from strategy.tier_runner import _realized_cash_curve
    eq = _realized_cash_curve(led, capital, dates)
    eq.index = pd.DatetimeIndex(dates)
    return eq


def _window_metrics(req: pd.Series, label: str) -> dict:
    """Sharpe / return / MaxDD of a realized-equity sub-window."""
    from strategy.tier_runner import _sharpe, _sortino, _max_dd
    if len(req) < 3:
        return {f"{label}_sharpe": float("nan"),
                f"{label}_ret_pct": float("nan"),
                f"{label}_max_dd": float("nan")}
    ret = req.pct_change().fillna(0.0)
    return {
        f"{label}_sharpe": _sharpe(ret),
        f"{label}_sortino": _sortino(ret),
        f"{label}_ret_pct": float(req.iloc[-1] / req.iloc[0] - 1.0) * 100.0,
        f"{label}_max_dd": _max_dd(req),
    }


def _concentrations(led: pd.DataFrame) -> dict:
    out = {"day_concentration": float("nan"),
           "ticker_concentration": float("nan")}
    if led is None or led.empty:
        return out
    pos = led[led["realized_pnl"] > 0]
    tot_pos = pos["realized_pnl"].sum()
    if tot_pos > 0:
        by_day = pos.groupby("close_date")["realized_pnl"].sum()
        out["day_concentration"] = float(by_day.max() / tot_pos)
        by_tk = pos.groupby("ticker")["realized_pnl"].sum()
        out["ticker_concentration"] = float(by_tk.max() / tot_pos)
    return out


def run_cell(cell) -> dict:
    """Run one perturbation cell. Executed inside a worker process."""
    name, overrides, ivr_override, slip_mult = cell
    t0 = time.time()

    from backtest import wheel_engine as _we
    from backtest.wheel_engine import run_wheel
    from strategy.tiers import balanced_tier, _full_wheel
    from strategy.tier_runner import (
        compute_metrics, _apply_iv_rank_floor, _load_spy_close)

    data = _W["data"]
    tier = _full_wheel(balanced_tier())          # Tier2_Balanced_FW
    cfg = replace(tier.wheel_cfg, **overrides)
    ivr = tier.iv_rank_floor if ivr_override is None else ivr_override

    # Match tier_runner.run_tier order EXACTLY: universe filter on FULL iv,
    # then subset, then iv-rank floor.
    tickers = tier.universe_filter(
        data["universe"], data["fundamentals"], data["iv"], data["prices"])
    px = data["prices"][data["prices"]["ticker"].isin(tickers)].copy()
    iv = data["iv"][data["iv"]["ticker"].isin(tickers)].copy()
    iv = _apply_iv_rank_floor(iv, ivr)

    orig_slip = (_we.SLIPPAGE_FRAC, _we.SLIPPAGE_MIN_TICKS)
    _we.SLIPPAGE_FRAC = orig_slip[0] * slip_mult
    _we.SLIPPAGE_MIN_TICKS = orig_slip[1] * slip_mult
    try:
        result = run_wheel(cfg=cfg, prices=px, iv=iv, macro=data["macro"],
                           fundamentals=data["fundamentals"],
                           universe=data["universe"],
                           starting_cash=CAPITAL,
                           start=START, end=END, verbose=False)
    finally:
        _we.SLIPPAGE_FRAC, _we.SLIPPAGE_MIN_TICKS = orig_slip

    m = compute_metrics(result, CAPITAL, spy_close=_load_spy_close(data))

    eq_df = result["equity_curve"].sort_values("date").reset_index(drop=True)
    led = result["ledger"]
    dates = pd.DatetimeIndex(pd.to_datetime(eq_df["date"]))
    req = _realized_daily(led, dates, CAPITAL)

    # Calmar on realized cash
    m["realized_calmar"] = (m["realized_cagr"] / abs(m["realized_max_dd"])
                            if m.get("realized_max_dd") else float("nan"))
    m.update(_concentrations(led))

    # per-year + crisis-window stratification of the realized curve
    strat = {}
    for yr in range(2020, 2026):
        sub = req[(req.index >= f"{yr}-01-01") & (req.index <= f"{yr}-12-31")]
        strat.update(_window_metrics(sub, f"y{yr}"))
    for label, (s, e) in CRISIS_WINDOWS.items():
        sub = req[(req.index >= s) & (req.index <= e)]
        strat.update(_window_metrics(sub, label))

    return {
        "cell": name,
        "knob": KNOB_OF_CELL.get(name, "baseline" if name == "baseline" else "slippage"),
        "overrides": overrides, "iv_rank_floor": ivr, "slip_mult": slip_mult,
        "n_universe": len(tickers),
        "metrics": m, "strata": strat,
        "equity": eq_df, "ledger": led,
        "runtime_s": time.time() - t0,
    }


# ---------------- verdict logic (pre-registered) ----------------

def build_verdict(rows: pd.DataFrame) -> dict:
    base = rows[rows.cell == "baseline"].iloc[0]
    b_sh = base["realized_sharpe"]
    b_dd = base["realized_max_dd"]
    fragile_knobs = []
    knob_worst = {}
    for knob in ["delta", "dte", "profit_take", "roll_trigger",
                 "iv_rank_floor", "max_names"]:
        sub = rows[rows.knob == knob]
        if sub.empty:
            continue
        worst_sh = float(sub["realized_sharpe"].min())
        worst_dd = float(sub["realized_max_dd"].min())
        knob_worst[knob] = {"worst_realized_sharpe": worst_sh,
                            "worst_realized_max_dd": worst_dd,
                            "sharpe_ratio_vs_base": worst_sh / b_sh if b_sh else float("nan")}
        if worst_sh < 0.50 * b_sh or worst_dd < 2.0 * b_dd:
            fragile_knobs.append(knob)

    # slippage breakeven multiple via linear interpolation of realized CAGR
    slip = rows[rows.knob.isin(["slippage", "baseline"])].sort_values("slip_mult")
    mults = slip["slip_mult"].values.astype(float)
    cagrs = slip["realized_cagr"].values.astype(float)
    breakeven = float("inf")
    for i in range(1, len(mults)):
        if cagrs[i - 1] > 0 >= cagrs[i]:
            frac = cagrs[i - 1] / (cagrs[i - 1] - cagrs[i])
            breakeven = mults[i - 1] + frac * (mults[i] - mults[i - 1])
            break
    if np.isinf(breakeven) and len(cagrs) >= 2 and cagrs[-1] > 0:
        # extrapolate from last two points if still positive at max mult
        slope = (cagrs[-1] - cagrs[-2]) / (mults[-1] - mults[-2])
        if slope < 0:
            breakeven = mults[-1] - cagrs[-1] / slope

    robust = (not fragile_knobs) and breakeven >= 2.0
    return {
        "baseline_realized_sharpe": float(b_sh),
        "baseline_realized_max_dd": float(b_dd),
        "baseline_replication_ok": bool(abs(b_sh - 1.49) <= 0.05),
        "knob_worst": knob_worst,
        "fragile_knobs": fragile_knobs,
        "slippage_breakeven_mult": (None if np.isinf(breakeven)
                                    else float(breakeven)),
        "verdict": "ROBUST" if robust else
                   ("FRAGILE-" + ",".join(fragile_knobs)
                    if fragile_knobs else "FRAGILE-slippage"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/wheel_v8_robustness_v1")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cells", default=None,
                    help="comma list of cell names to run (default all)")
    args = ap.parse_args()
    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cells = CELLS
    if args.cells:
        want = set(args.cells.split(","))
        cells = [c for c in CELLS if c[0] in want]

    print(f"[robustness] running {len(cells)} cells, {args.workers} workers")
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_worker_init) as pool:
        futs = {pool.submit(run_cell, c): c[0] for c in cells}
        for f in as_completed(futs):
            r = f.result()
            results[r["cell"]] = r
            m = r["metrics"]
            print(f"[robustness] {r['cell']:10s} done in {r['runtime_s']:.0f}s  "
                  f"rSharpe={m['realized_sharpe']:.2f} rCAGR={m['realized_cagr']*100:.1f}% "
                  f"rDD={m['realized_max_dd']*100:.1f}% PF={m['pf']:.2f} "
                  f"WR={m['wr']*100:.1f}% gap={m.get('regime_gap', float('nan')):.2f}",
                  flush=True)
            r["equity"].to_parquet(out / f"equity_{r['cell']}.parquet", index=False)
            r["ledger"].to_parquet(out / f"ledger_{r['cell']}.parquet", index=False)

    # flat table
    rows = []
    for name, r in results.items():
        rows.append({"cell": name, "knob": r["knob"],
                     "slip_mult": r["slip_mult"],
                     "iv_rank_floor": r["iv_rank_floor"],
                     "n_universe": r["n_universe"],
                     **{k: v for k, v in r["overrides"].items()},
                     **r["metrics"], **r["strata"],
                     "runtime_s": r["runtime_s"]})
    df = pd.DataFrame(rows)
    order = [c[0] for c in CELLS if c[0] in set(df.cell)]
    df = df.set_index("cell").loc[order].reset_index()
    df.to_csv(out / "sensitivity_results.csv", index=False)
    df.to_parquet(out / "sensitivity_results.parquet", index=False)

    verdict = build_verdict(df) if "baseline" in set(df.cell) else {"verdict": "INCOMPLETE"}
    summary = {
        "experiment": "wheel_v8_robustness_v1",
        "config_tested": "Tier2_Balanced_FW (v8_WF: WF skew, slippage, regime overlay, real-blend IV)",
        "window": [START, END], "capital": CAPITAL,
        "cells_run": sorted(results.keys()),
        "preregistered_rules": {
            "fragile_if": "realized Sharpe < 0.50x baseline OR realized MaxDD < 2x baseline",
            "robust_if": "no fragile knob AND slippage breakeven >= 2.0x",
            "baseline_replication_tolerance": 0.05,
        },
        **verdict,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(verdict, indent=2, default=str))

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("wheel_v8_robustness_v1")
        for name in order:
            row = df[df.cell == name].iloc[0]
            with mlflow.start_run(run_name=name):
                mlflow.log_params({"cell": name, "knob": row["knob"],
                                   "slip_mult": row["slip_mult"],
                                   "iv_rank_floor": row["iv_rank_floor"],
                                   "config": "Tier2_Balanced_FW_v8WF"})
                for k in ["realized_sharpe", "realized_sortino", "realized_cagr",
                          "realized_max_dd", "realized_calmar", "pf", "wr",
                          "sharpe", "sortino", "cagr", "max_dd",
                          "regime_gap", "regime_green_sharpe", "regime_red_sharpe",
                          "day_concentration", "ticker_concentration",
                          "covid_2020_max_dd", "bear_2022_max_dd",
                          "unwind_aug2024_max_dd"]:
                    v = row.get(k)
                    if v is not None and np.isfinite(float(v)):
                        mlflow.log_metric(k, float(v))
        with mlflow.start_run(run_name="SUMMARY"):
            mlflow.set_tag("verdict", summary.get("verdict", "?"))
            mlflow.log_artifact(str(out / "summary.json"))
            mlflow.log_artifact(str(out / "sensitivity_results.csv"))
        print("[robustness] MLflow logged -> wheel_v8_robustness_v1")
    except Exception as e:
        print(f"[robustness] MLflow logging failed (non-fatal): {e}",
              file=sys.stderr)

    print(f"[robustness] DONE -> {out}")


if __name__ == "__main__":
    main()
