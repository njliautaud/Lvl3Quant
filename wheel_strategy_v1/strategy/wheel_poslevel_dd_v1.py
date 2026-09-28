"""
wheel_poslevel_dd_v1.py — POSITION-LEVEL drawdown levers + O2 vol-scaling
robustness for the canonical v8_WF wheel config (Tier2_Balanced_FW).
Follow-up to wheel_v8_robustness_v1 (ROBUST, realistic forward DD ~-15%)
and wheel_dd_overlay_v1 (NEGATIVE: entry-side overlays cannot cap DD; the
tail lives in positions opened BEFORE vol spikes — assignment/roll losses).

PRE-REGISTERED DESIGN (written before any cell is run — 2026-06-11):

ENGINE-FLAG SEMANTICS (verified in backtest/wheel_engine.py before design;
flags are dormant defaults — engine file and live paper engines UNTOUCHED,
all runs go through the replication-verified harness copy in
strategy/wheel_dd_overlay_v1.run_wheel_overlay which contains identical
flag code paths and whose no-flag/no-overlay path is byte-identical):

  max_assigned_notional_pct (default 1.0 = off):
    If MV of held shares (long_shares + short_call legs) > cap * equity,
    BLOCK new short puts that day. IMPORTANT HONEST NOTE: contrary to the
    task brief's assumption, this flag does NOT liquidate excess shares —
    it only gates NEW entries. It is therefore an entry-side mechanism
    (same family that failed in wheel_dd_overlay_v1), tested here anyway
    per pre-registration because it gates on ASSIGNED EXPOSURE rather
    than equity/vol signals.
  share_stop_loss_pct (default 0.0 = off):
    Daily: for any assigned position (long_shares or short_call), if
    close < share_cost_basis * (1 - x): buy back the open CC (forced_close)
    and sell all shares at close (exit_reason stop_loss). This is the true
    POSITION-LEVEL lever — it acts on already-open positions.

PART A — position-level DD levers. Variants (cfg-flag overrides only):
  noop   : both flags at defaults — must replicate wheel_v8_robustness_v1
           (baseline rSharpe 1.4915 +/-0.05; all 5 config DDs +/-0.01).
  cap25  : max_assigned_notional_pct=0.25
  cap40  : max_assigned_notional_pct=0.40
  stop8  : share_stop_loss_pct=0.08
  stop12 : share_stop_loss_pct=0.12
  stop15 : share_stop_loss_pct=0.15
Configs (5): v8_WF baseline + 4 high-DD neighbors (dte_24_36, roll_3,
pt_078, ivr_030). 30 single cells.
COMBO RULE (pre-registered): after singles, pick best_stop and best_cap =
the level maximizing worst-neighbor realized MaxDD subject to baseline
sharpe_cost < 10% and baseline regime gap <= 0.50. Run the single combo
(best_cap x best_stop) on all 5 configs ONLY IF at least one single
variant lifts worst-neighbor DD above -0.14 (>= 2pp improvement vs the
-0.160 no-flag worst). Otherwise skip (5 cells max).
PART A GATE (identical to wheel_dd_overlay_v1):
  SUCCESS: worst neighbor rDD > -0.10 AND baseline sharpe_cost < 10%
           AND baseline regime gap <= 0.50
  PARTIAL: worst neighbor rDD > -0.12 with same Sharpe/gap gates
  FAIL   : otherwise. Honest negative acceptable.

PART B — O2 vol-scaling robustness (baseline config ONLY, 8 cells).
O2 generalized: p_t = expanding past-only percentile (>=252 prior obs,
SPY from 2015) of SPY {lb}d realized vol; scale = 1 below lo pct, s1 in
[lo, hi), s2 at >= hi. New-position exposure scaling identical to
wheel_dd_overlay_v1 (eff max names + per-name alloc scaling).
  o2_orig          : lo=0.70 hi=0.90 s1=0.50 s2=0.25 lb=20  (replication
                     cell — must match 1.7441 +/-0.05)
  o2_bp6085        : lo=0.60 hi=0.85 (orig scales, lb20)
  o2_bp7595        : lo=0.75 hi=0.95
  o2_sc7040        : s1=0.70 s2=0.40 (orig bps, lb20)
  o2_lb10          : lb=10 (orig bps/scales)
  o2_lb30          : lb=30
  o2_bp6085_sc7040 : lo=0.60 hi=0.85 s1=0.70 s2=0.40
  o2_bp7595_sc7040 : lo=0.75 hi=0.95 s1=0.70 s2=0.40
PART B VERDICT (pre-registered, on baseline realized Sharpe across all 8
O2 cells):
  PLATEAU        : min Sharpe >= 1.60 AND all regime gaps <= 0.50 AND all
                   rDD better than -0.10 -> promotion candidate (Sharpe
                   improver, separate from the DD question).
  THRESHOLD_LUCK : any cell < 1.4915 (worse than no overlay) -> reject.
  MIXED          : otherwise (report honestly, no promotion).

Cell budget: 30 + (<=5 combo) + 8 = <=43 (cap 45).

Usage (from /home/jupiter/Lvl3Quant/wheel_strategy_v1):
    python3 -m strategy.wheel_poslevel_dd_v1 \
        --out results/wheel_poslevel_dd_v1 --workers 6
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

START = "2020-01-01"
END = "2025-12-31"
CAPITAL = 100_000.0

CRISIS_WINDOWS = {
    "covid_2020": ("2020-02-19", "2020-04-30"),
    "bear_2022": ("2022-01-03", "2022-10-14"),
    "unwind_aug2024": ("2024-07-15", "2024-08-15"),
}

ROBUSTNESS_NONE_DD = {
    "baseline": -0.08350859213087147,
    "dte_24_36": -0.1555774581657864,
    "roll_3": -0.16001770797531634,
    "pt_078": -0.14062860750669648,
    "ivr_030": -0.1381404795509511,
}
BASELINE_SHARPE_REF = 1.491544910957763
O2_ORIG_SHARPE_REF = 1.7440598813834312
WORST_NONE_DD = -0.160
NEIGHBORS = ["dte_24_36", "roll_3", "pt_078", "ivr_030"]

CONFIGS = [
    ("baseline",  {},                              None),
    ("dte_24_36", {"dte_min": 24, "dte_max": 36},  None),
    ("roll_3",    {"roll_dte_trigger": 3},         None),
    ("pt_078",    {"profit_take_pct": 0.78},       None),
    ("ivr_030",   {},                              0.30),
]

# Part A variants: (name, flag overrides applied via dataclasses.replace)
PARTA_VARIANTS = [
    ("noop",   {}),
    ("cap25",  {"max_assigned_notional_pct": 0.25}),
    ("cap40",  {"max_assigned_notional_pct": 0.40}),
    ("stop8",  {"share_stop_loss_pct": 0.08}),
    ("stop12", {"share_stop_loss_pct": 0.12}),
    ("stop15", {"share_stop_loss_pct": 0.15}),
]

# Part B O2 perturbations: (name, lo, hi, s1, s2, lookback)
PARTB_CELLS = [
    ("o2_orig",          0.70, 0.90, 0.50, 0.25, 20),
    ("o2_bp6085",        0.60, 0.85, 0.50, 0.25, 20),
    ("o2_bp7595",        0.75, 0.95, 0.50, 0.25, 20),
    ("o2_sc7040",        0.70, 0.90, 0.70, 0.40, 20),
    ("o2_lb10",          0.70, 0.90, 0.50, 0.25, 10),
    ("o2_lb30",          0.70, 0.90, 0.50, 0.25, 30),
    ("o2_bp6085_sc7040", 0.60, 0.85, 0.70, 0.40, 20),
    ("o2_bp7595_sc7040", 0.75, 0.95, 0.70, 0.40, 20),
]


# ---------------- parameterized O2 signal + scale ----------------

def build_spy_signals_lb(spy_close: pd.Series, lookback: int) -> pd.DataFrame:
    """Same as wheel_dd_overlay_v1.build_spy_signals but with a
    parameterized realized-vol lookback. Past-only expanding percentile."""
    s = spy_close.sort_index()
    ret = s.pct_change()
    vol = ret.rolling(lookback).std() * np.sqrt(252)
    v = vol.values
    pct = np.full(len(v), np.nan)
    hist = []
    for i, x in enumerate(v):
        if np.isfinite(x):
            if len(hist) >= 252:
                pct[i] = np.mean(np.array(hist) <= x)
            hist.append(x)
    ma200 = s.rolling(200).mean()
    return pd.DataFrame({"vol_pct": pct,
                         "below_200ma": (s < ma200).values}, index=s.index)


def overlay_scale_ext(spec, dt, equity_hist, sig):
    """Extension of wheel_dd_overlay_v1.overlay_scale adding kind 'o2p'
    (parameterized breakpoints/scales). Installed via module attribute so
    run_wheel_overlay picks it up; delegates everything else unchanged."""
    from strategy import wheel_dd_overlay_v1 as ddov
    if spec is not None and spec.get("kind") == "o2p":
        if sig is None or dt not in sig.index:
            return 1.0
        p = sig.loc[dt, "vol_pct"]
        if not np.isfinite(p):
            return 1.0
        if p >= spec["hi"]:
            return spec["s2"]
        if p >= spec["lo"]:
            return spec["s1"]
        return 1.0
    return ddov._overlay_scale_orig(spec, dt, equity_hist, sig)


# ---------------- per-process worker state ----------------
_W = {}


def _worker_init():
    from strategy import iv_skew as _ivs
    from strategy import wheel_dd_overlay_v1 as ddov
    from strategy.tier_runner import _load_inputs, _load_spy_close
    from strategy.regime_overlay import build_regime, apply_regime_gate

    # install the extended scale function (process-local; the overlay
    # module on disk is untouched)
    if not hasattr(ddov, "_overlay_scale_orig"):
        ddov._overlay_scale_orig = ddov.overlay_scale
        ddov.overlay_scale = overlay_scale_ext

    _ivs.load_calibration_walkforward()
    data = _load_inputs(modeled=False, smoke=False, real_iv=True)
    regime = build_regime()
    data["macro"] = apply_regime_gate(data["macro"], regime,
                                      vix_force_gate=999.0)
    _W["data"] = data
    spy = _load_spy_close(data)
    _W["spy_close"] = spy
    _W["spy_signals"] = {}
    if spy is not None:
        for lb in sorted({c[5] for c in PARTB_CELLS}):
            _W["spy_signals"][lb] = build_spy_signals_lb(spy, lb)


def _window_metrics(req: pd.Series, label: str) -> dict:
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
    """cell = (part, cell_name, cfg_name, cfg_overrides, ivr_override,
               flag_overrides, ov_spec, lookback)"""
    (part, cell_name, cfg_name, overrides, ivr_override,
     flag_overrides, ov_spec, lookback) = cell
    t0 = time.time()

    from strategy.tiers import balanced_tier, _full_wheel
    from strategy.tier_runner import (
        compute_metrics, _apply_iv_rank_floor, _realized_cash_curve)
    from strategy.wheel_dd_overlay_v1 import run_wheel_overlay

    data = _W["data"]
    tier = _full_wheel(balanced_tier())          # Tier2_Balanced_FW
    cfg = replace(tier.wheel_cfg, **overrides, **flag_overrides)
    ivr = tier.iv_rank_floor if ivr_override is None else ivr_override

    tickers = tier.universe_filter(
        data["universe"], data["fundamentals"], data["iv"], data["prices"])
    px = data["prices"][data["prices"]["ticker"].isin(tickers)].copy()
    iv = data["iv"][data["iv"]["ticker"].isin(tickers)].copy()
    iv = _apply_iv_rank_floor(iv, ivr)

    sig = _W["spy_signals"].get(lookback) if ov_spec is not None else None
    result = run_wheel_overlay(
        cfg=cfg, prices=px, iv=iv, macro=data["macro"],
        fundamentals=data["fundamentals"], universe=data["universe"],
        starting_cash=CAPITAL, start=START, end=END,
        overlay_spec=ov_spec, spy_signals=sig)

    m = compute_metrics(result, CAPITAL, spy_close=_W["spy_close"])

    eq_df = result["equity_curve"].sort_values("date").reset_index(drop=True)
    led = result["ledger"]
    dates = pd.DatetimeIndex(pd.to_datetime(eq_df["date"]))
    req = _realized_cash_curve(led, CAPITAL, dates)
    req.index = dates

    m["realized_calmar"] = (m["realized_cagr"] / abs(m["realized_max_dd"])
                            if m.get("realized_max_dd") else float("nan"))
    m.update(_concentrations(led))
    if led is not None and not led.empty and "exit_reason" in led.columns:
        m["n_stop_loss_exits"] = int((led["exit_reason"] == "stop_loss").sum())
        stop_led = led[led["exit_reason"] == "stop_loss"]
        m["stop_loss_realized_pnl"] = float(stop_led["realized_pnl"].sum())
    else:
        m["n_stop_loss_exits"] = 0
        m["stop_loss_realized_pnl"] = 0.0
    sc = result["scale_curve"]
    m["pct_days_scaled"] = float((sc["scale"] < 1.0).mean()) if len(sc) else 0.0

    strat = {}
    for yr in range(2020, 2026):
        sub = req[(req.index >= f"{yr}-01-01") & (req.index <= f"{yr}-12-31")]
        strat.update(_window_metrics(sub, f"y{yr}"))
    for label, (s, e) in CRISIS_WINDOWS.items():
        sub = req[(req.index >= s) & (req.index <= e)]
        strat.update(_window_metrics(sub, label))

    return {
        "cell": cell_name, "part": part, "config": cfg_name,
        "variant": cell_name.split("__", 1)[1],
        "overrides": overrides, "flag_overrides": flag_overrides,
        "iv_rank_floor": ivr, "n_universe": len(tickers),
        "metrics": m, "strata": strat,
        "equity": eq_df, "ledger": led,
        "runtime_s": time.time() - t0,
    }


# ---------------- cell construction ----------------

def parta_cells(variants):
    cells = []
    for cfg_name, overrides, ivr in CONFIGS:
        for vn, flags in variants:
            cells.append(("A", f"{cfg_name}__{vn}", cfg_name, overrides,
                          ivr, flags, None, None))
    return cells


def partb_cells():
    cells = []
    cfg_name, overrides, ivr = CONFIGS[0]   # baseline only
    for name, lo, hi, s1, s2, lb in PARTB_CELLS:
        spec = {"kind": "o2p", "lo": lo, "hi": hi, "s1": s1, "s2": s2}
        cells.append(("B", f"{cfg_name}__{name}", cfg_name, overrides,
                      ivr, {}, spec, lb))
    return cells


# ---------------- verdict logic (pre-registered) ----------------

def build_verdicts(df: pd.DataFrame) -> dict:
    def get(cfg, var, col):
        sub = df[(df.config == cfg) & (df.variant == var)]
        return float(sub.iloc[0][col]) if len(sub) else float("nan")

    base_sh = get("baseline", "noop", "realized_sharpe")
    replication_ok = bool(abs(base_sh - BASELINE_SHARPE_REF) <= 0.05)
    neighbor_repl = {}
    for cfg, ref_dd in ROBUSTNESS_NONE_DD.items():
        dd = get(cfg, "noop", "realized_max_dd")
        neighbor_repl[cfg] = {
            "dd": dd, "ref": ref_dd,
            "ok": bool(np.isfinite(dd) and abs(dd - ref_dd) <= 0.01)}

    # ---- Part A ----
    a_vars = sorted(v for v in df[df.part == "A"].variant.unique()
                    if v != "noop")
    per_variant = {}
    for vn in a_vars:
        b_sh = get("baseline", vn, "realized_sharpe")
        sharpe_cost = 1.0 - (b_sh / base_sh) if base_sh else float("nan")
        gap = get("baseline", vn, "regime_gap")
        n_dds = [get(c, vn, "realized_max_dd") for c in NEIGHBORS]
        n_dds = [d for d in n_dds if np.isfinite(d)]
        worst = min(n_dds) if n_dds else float("nan")
        success = (np.isfinite(worst) and worst > -0.10
                   and sharpe_cost < 0.10
                   and np.isfinite(gap) and gap <= 0.50)
        partial = (not success and np.isfinite(worst) and worst > -0.120
                   and sharpe_cost < 0.10 and np.isfinite(gap) and gap <= 0.50)
        per_variant[vn] = {
            "baseline_sharpe": b_sh,
            "baseline_cagr": get("baseline", vn, "realized_cagr"),
            "baseline_max_dd": get("baseline", vn, "realized_max_dd"),
            "sharpe_cost_pct": sharpe_cost * 100.0,
            "regime_gap_baseline": gap,
            "worst_neighbor_dd": worst,
            "neighbor_dds": {c: get(c, vn, "realized_max_dd")
                             for c in NEIGHBORS},
            "verdict": ("SUCCESS" if success else
                        "PARTIAL" if partial else "FAIL"),
        }
    a_overall = ("SUCCESS" if any(v["verdict"] == "SUCCESS"
                                  for v in per_variant.values()) else
                 "PARTIAL" if any(v["verdict"] == "PARTIAL"
                                  for v in per_variant.values()) else "FAIL")

    # ---- Part B ----
    b_rows = df[df.part == "B"]
    per_o2 = {}
    for _, r in b_rows.iterrows():
        per_o2[r["variant"]] = {
            "sharpe": float(r["realized_sharpe"]),
            "cagr": float(r["realized_cagr"]),
            "max_dd": float(r["realized_max_dd"]),
            "regime_gap": float(r["regime_gap"]),
            "y2022_sharpe": float(r.get("y2022_sharpe", float("nan"))),
            "pct_days_scaled": float(r["pct_days_scaled"]),
        }
    b_sh_vals = [v["sharpe"] for v in per_o2.values() if np.isfinite(v["sharpe"])]
    o2_orig_sh = per_o2.get("o2_orig", {}).get("sharpe", float("nan"))
    o2_repl_ok = bool(np.isfinite(o2_orig_sh)
                      and abs(o2_orig_sh - O2_ORIG_SHARPE_REF) <= 0.05)
    min_sh = min(b_sh_vals) if b_sh_vals else float("nan")
    gaps_ok = all(np.isfinite(v["regime_gap"]) and v["regime_gap"] <= 0.50
                  for v in per_o2.values())
    dds_ok = all(np.isfinite(v["max_dd"]) and v["max_dd"] > -0.10
                 for v in per_o2.values())
    if np.isfinite(min_sh) and min_sh < BASELINE_SHARPE_REF:
        b_verdict = "THRESHOLD_LUCK"
    elif np.isfinite(min_sh) and min_sh >= 1.60 and gaps_ok and dds_ok:
        b_verdict = "PLATEAU"
    else:
        b_verdict = "MIXED"

    return {
        "baseline_realized_sharpe": base_sh,
        "baseline_replication_ok": replication_ok,
        "neighbor_replication": neighbor_repl,
        "worst_neighbor_dd_no_flags": WORST_NONE_DD,
        "part_a": {"per_variant": per_variant, "verdict": a_overall},
        "part_b": {"o2_orig_replication_ok": o2_repl_ok,
                   "per_cell": per_o2, "min_sharpe": min_sh,
                   "all_gaps_le_050": gaps_ok, "all_dd_better_than_-10": dds_ok,
                   "verdict": b_verdict},
    }


def combo_choice(df: pd.DataFrame):
    """Pre-registered combo rule. Returns (cap_level, stop_level) or None."""
    def get(cfg, var, col):
        sub = df[(df.config == cfg) & (df.variant == var)]
        return float(sub.iloc[0][col]) if len(sub) else float("nan")

    base_sh = get("baseline", "noop", "realized_sharpe")

    def stats(vn):
        b_sh = get("baseline", vn, "realized_sharpe")
        cost = 1.0 - (b_sh / base_sh) if base_sh else float("nan")
        gap = get("baseline", vn, "regime_gap")
        dds = [get(c, vn, "realized_max_dd") for c in NEIGHBORS]
        dds = [d for d in dds if np.isfinite(d)]
        worst = min(dds) if dds else float("nan")
        return worst, cost, gap

    caps = {"cap25": 0.25, "cap40": 0.40}
    stops = {"stop8": 0.08, "stop12": 0.12, "stop15": 0.15}

    def best(group):
        cands = []
        for vn, lvl in group.items():
            worst, cost, gap = stats(vn)
            if (np.isfinite(worst) and cost < 0.10
                    and np.isfinite(gap) and gap <= 0.50):
                cands.append((worst, vn, lvl))
        return max(cands) if cands else None

    bc, bs = best(caps), best(stops)
    trigger = any(x is not None and x[0] > -0.14 for x in (bc, bs))
    if not trigger:
        return None
    cap_lvl = bc[2] if bc else 1.0
    stop_lvl = bs[2] if bs else 0.0
    if cap_lvl >= 1.0 or stop_lvl <= 0.0:
        return None  # combo needs both arms to have a qualifying level
    return cap_lvl, stop_lvl


# ---------------- main ----------------

def _run_pool(cells, workers, out, results):
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_worker_init) as pool:
        futs = {pool.submit(run_cell, c): c for c in cells}
        for f in as_completed(futs):
            r = f.result()
            results[r["cell"]] = r
            m = r["metrics"]
            print(f"[poslevel] {r['cell']:30s} done {r['runtime_s']:5.0f}s  "
                  f"rSharpe={m['realized_sharpe']:.2f} "
                  f"rCAGR={m['realized_cagr']*100:5.1f}% "
                  f"rDD={m['realized_max_dd']*100:5.1f}% "
                  f"gap={m.get('regime_gap', float('nan')):.2f} "
                  f"stops={m['n_stop_loss_exits']}", flush=True)
            r["equity"].to_parquet(out / f"equity_{r['cell']}.parquet",
                                   index=False)
            r["ledger"].to_parquet(out / f"ledger_{r['cell']}.parquet",
                                   index=False)


def _to_df(results):
    rows = []
    for name, r in results.items():
        rows.append({"cell": name, "part": r["part"], "config": r["config"],
                     "variant": r["variant"],
                     "iv_rank_floor": r["iv_rank_floor"],
                     "n_universe": r["n_universe"],
                     **r["metrics"], **r["strata"],
                     "runtime_s": r["runtime_s"]})
    return (pd.DataFrame(rows)
            .sort_values(["part", "config", "variant"])
            .reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/wheel_poslevel_dd_v1")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cells = parta_cells(PARTA_VARIANTS) + partb_cells()
    assert len(cells) <= 45, f"cell cap exceeded: {len(cells)}"
    print(f"[poslevel] phase 1: {len(cells)} cells, {args.workers} workers")
    results = {}
    _run_pool(cells, args.workers, out, results)

    df = _to_df(results)
    choice = combo_choice(df)
    if choice is not None:
        cap_lvl, stop_lvl = choice
        vn = f"combo_cap{int(cap_lvl*100)}_stop{int(stop_lvl*100)}"
        print(f"[poslevel] combo triggered: {vn}")
        combo = parta_cells([(vn, {"max_assigned_notional_pct": cap_lvl,
                                   "share_stop_loss_pct": stop_lvl})])
        assert len(results) + len(combo) <= 45
        _run_pool(combo, args.workers, out, results)
        df = _to_df(results)
    else:
        vn = None
        print("[poslevel] combo NOT triggered (no single lifted "
              "worst-neighbor DD above -0.14 with qualifying cost/gap)")

    df.to_csv(out / "poslevel_results.csv", index=False)
    df.to_parquet(out / "poslevel_results.parquet", index=False)

    verdict = build_verdicts(df)
    summary = {
        "experiment": "wheel_poslevel_dd_v1",
        "config_tested": ("Tier2_Balanced_FW (v8_WF) + position-level engine "
                          "flags (Part A) + O2 vol-scaling perturbations (Part B)"),
        "window": [START, END], "capital": CAPITAL,
        "cells_run": sorted(results.keys()),
        "combo_run": vn,
        "flag_semantics_note": (
            "max_assigned_notional_pct BLOCKS new CSPs while assigned-share "
            "MV > cap*equity (entry gate, does NOT liquidate); "
            "share_stop_loss_pct force-liquidates assigned shares (+CC "
            "buyback) when close < basis*(1-x) (true position-level exit)."),
        "preregistered_rules": {
            "part_a_success": ("worst neighbor rDD > -10% AND baseline Sharpe "
                               "cost < 10% AND baseline regime gap <= 0.50"),
            "part_a_partial": "worst neighbor rDD > -12% same gates",
            "combo_rule": ("best cap x best stop on all configs IFF some "
                           "single lifts worst-neighbor DD above -0.14 with "
                           "cost<10% and gap<=0.50"),
            "part_b_plateau": ("min O2-cell Sharpe >= 1.60 AND gaps <= 0.50 "
                               "AND DDs > -10%"),
            "part_b_luck": "any O2 cell Sharpe < 1.4915 (no-overlay baseline)",
            "baseline_replication_tolerance": 0.05,
        },
        **verdict,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2,
                                                 default=str))
    print(json.dumps(verdict, indent=2, default=str))

    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("wheel_poslevel_dd_v1")
        for _, row in df.iterrows():
            with mlflow.start_run(run_name=row["cell"]):
                mlflow.log_params({"part": row["part"],
                                   "config": row["config"],
                                   "variant": row["variant"],
                                   "iv_rank_floor": row["iv_rank_floor"],
                                   "base": "Tier2_Balanced_FW_v8WF"})
                for k in ["realized_sharpe", "realized_sortino", "realized_cagr",
                          "realized_max_dd", "realized_calmar", "pf", "wr",
                          "regime_gap", "regime_green_sharpe",
                          "regime_red_sharpe", "day_concentration",
                          "ticker_concentration", "n_stop_loss_exits",
                          "stop_loss_realized_pnl", "pct_days_scaled",
                          "covid_2020_max_dd", "bear_2022_max_dd",
                          "unwind_aug2024_max_dd",
                          "y2022_sharpe"]:
                    v = row.get(k)
                    try:
                        if v is not None and np.isfinite(float(v)):
                            mlflow.log_metric(k, float(v))
                    except (TypeError, ValueError):
                        pass
        with mlflow.start_run(run_name="SUMMARY"):
            mlflow.set_tag("part_a_verdict", verdict["part_a"]["verdict"])
            mlflow.set_tag("part_b_verdict", verdict["part_b"]["verdict"])
            mlflow.log_artifact(str(out / "summary.json"))
            mlflow.log_artifact(str(out / "poslevel_results.csv"))
        print("[poslevel] MLflow logged -> wheel_poslevel_dd_v1")
    except Exception as e:
        print(f"[poslevel] MLflow logging failed (non-fatal): {e}",
              file=sys.stderr)

    print(f"[poslevel] DONE -> {out}")


if __name__ == "__main__":
    main()
