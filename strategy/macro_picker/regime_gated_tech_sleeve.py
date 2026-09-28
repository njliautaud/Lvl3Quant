"""
Regime-gated tech sleeve on top of the leader book.

Hypothesis: the 60/40 tech+leader blend FAILED HC #428 R1 gate (Sharpe gap
green/red = 1.62, ceiling 0.50) because tech rotation is a long-beta vehicle.
If we only deploy the tech sleeve when the SPY regime is "ok" (uptrend), and
fall back to leader-only on red/weak days, the combined book may pass.

Regime signals tested (all lagged 1d to avoid look-ahead):
  A: SPY 20-day return > 0
  B: SPY 50-day SMA slope > 0
  C: SPY above 200-day SMA
  D: SPY 5-day return > 0

For each signal:
  regime_ok day  -> X% tech + (1-X)% leader,  X in {0.2, 0.3, 0.4, 0.5, 0.6}
  regime_off day -> 100% leader

Metrics: CAGR, Sharpe, Sortino, Calmar, MaxDD, PF, WR.
HC #428 R1 gate: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) <= 0.50
Day concentration: top-1 / total positive -> <= 0.70.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
LEADER_BOOK = ROOT / "output/macro_picker/etf_rotation_regime_20260608_164726_hold21_longonly/book.parquet"
TECH_BOOK = ROOT / "output/macro_picker/tech_sub_industry_rotation_20260609_072759/book.parquet"
SPY_PRICE_PATH = ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet"

OUT_DIR = ROOT / f"output/macro_picker/regime_gated_tech_{time.strftime('%Y%m%d_%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = [0.2, 0.3, 0.4, 0.5, 0.6]
SIGNALS = ["A_ret20", "B_sma50slope", "C_above_sma200", "D_ret5"]


def _load_book(path: Path, label: str) -> pd.Series:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["daily_ret"].astype(float).sort_index()
    s.name = label
    return s


def _load_spy_close() -> pd.Series:
    px = pd.read_parquet(SPY_PRICE_PATH)
    px["date"] = pd.to_datetime(px["date"])
    spy = px[px["ticker"] == "SPY"].sort_values("date").set_index("date")["close"].astype(float)
    return spy


def build_regime_signals(spy: pd.Series) -> pd.DataFrame:
    """Build 4 regime booleans. All are computed on closes up to date t-1 so the
    signal used to size positions on date t never peeks at date-t close.

    Returned DataFrame is indexed by date and has bool columns for each signal.
    Value at index t = signal evaluated using data through t-1.
    """
    df = pd.DataFrame(index=spy.index)
    # A: 20-day return > 0 -> sign of close_{t-1} / close_{t-21} - 1.
    ret20 = spy.pct_change(20)
    df["A_ret20"] = (ret20.shift(1) > 0)
    # B: 50-day SMA slope (today - yesterday's SMA) > 0
    sma50 = spy.rolling(50, min_periods=50).mean()
    df["B_sma50slope"] = (sma50.shift(1).diff() > 0)
    # C: SPY > 200-day SMA
    sma200 = spy.rolling(200, min_periods=200).mean()
    df["C_above_sma200"] = (spy.shift(1) > sma200.shift(1))
    # D: 5-day return > 0
    ret5 = spy.pct_change(5)
    df["D_ret5"] = (ret5.shift(1) > 0)
    return df.fillna(False)


def metrics(daily: pd.Series) -> dict:
    """Standard risk-adjusted metrics + day-concentration."""
    daily = daily.dropna().sort_index()
    if daily.empty or len(daily) < 5:
        return {"n_days": int(len(daily))}
    eq = (1.0 + daily).cumprod()
    mu = daily.mean()
    sd = daily.std()
    sharpe = (mu / sd) * np.sqrt(252) if sd > 0 else 0.0
    down = daily[daily < 0].std()
    sortino = (mu / down) * np.sqrt(252) if down and down > 0 else 0.0
    n_cal_days = max(1, (daily.index[-1] - daily.index[0]).days)
    cagr = float(eq.iloc[-1]) ** (365.25 / n_cal_days) - 1.0
    roll = eq.cummax()
    dd = (eq - roll) / roll
    mdd = float(dd.min())
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0
    wr = float((daily > 0).mean())
    gains = daily[daily > 0].sum()
    losses = -daily[daily < 0].sum()
    pf = float(gains / losses) if losses > 0 else 0.0
    pos = daily[daily > 0]
    day_conc = float(pos.max() / pos.sum()) if pos.sum() > 0 else 0.0
    return {
        "n_days": int(len(daily)),
        "cagr_pct": round(cagr * 100, 3),
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "calmar": round(float(calmar), 4),
        "max_dd_pct": round(mdd * 100, 3),
        "pf": round(float(pf), 4),
        "wr": round(float(wr), 4),
        "day_conc": round(float(day_conc), 4),
    }


def classify_regime_by_spy(spy: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """green/red/flat by SPY close-to-close return on the same date."""
    ret = spy.pct_change()
    out = pd.Series(index=idx, dtype="object")
    for d in idx:
        r = ret.get(d, np.nan)
        if pd.isna(r):
            out.loc[d] = "flat"
        elif r > 0.001:
            out.loc[d] = "green"
        elif r < -0.001:
            out.loc[d] = "red"
        else:
            out.loc[d] = "flat"
    return out


def stratified_sharpe(daily: pd.Series, spy: pd.Series) -> dict:
    cls = classify_regime_by_spy(spy, daily.index)
    out = {}
    for label in ["green", "red", "flat"]:
        sub = daily[cls == label]
        if len(sub) >= 5 and sub.std() > 0:
            sr = float((sub.mean() / sub.std()) * np.sqrt(252))
        else:
            sr = 0.0
        out[f"sharpe_{label}"] = round(sr, 4)
        out[f"n_{label}"] = int(len(sub))
    sg = out.get("sharpe_green", 0.0)
    sr = out.get("sharpe_red", 0.0)
    denom = max(abs(sg), abs(sr), 1e-9)
    gap = abs(sg - sr) / denom
    out["regime_gap"] = round(float(gap), 4)
    out["regime_gate_pass"] = bool(gap <= 0.50)
    return out


def build_gated_book(leader: pd.Series, tech: pd.Series, regime_on: pd.Series,
                     w_tech: float) -> pd.Series:
    """When regime_on is True for date t, blend = w_tech*tech + (1-w_tech)*leader.
    When False, blend = leader only.
    """
    common = leader.index.intersection(tech.index).intersection(regime_on.index)
    L = leader.reindex(common)
    T = tech.reindex(common)
    R = regime_on.reindex(common).fillna(False).astype(bool)
    blended = pd.Series(index=common, dtype=float)
    blended[R] = w_tech * T[R] + (1.0 - w_tech) * L[R]
    blended[~R] = L[~R]
    return blended.dropna()


def main() -> None:
    leader = _load_book(LEADER_BOOK, "leader")
    tech = _load_book(TECH_BOOK, "tech")
    spy = _load_spy_close()
    regimes = build_regime_signals(spy)

    common = leader.index.intersection(tech.index)
    overlap_start = common.min().date().isoformat()
    overlap_end = common.max().date().isoformat()
    n_overlap = len(common)

    # Baseline: leader-only on the overlap
    leader_only = leader.reindex(common).dropna()
    baseline_m = metrics(leader_only)
    baseline_strat = stratified_sharpe(leader_only, spy)
    baseline = {**baseline_m, **baseline_strat, "label": "leader_only_baseline"}

    # Pure tech reference too
    tech_only = tech.reindex(common).dropna()
    tech_m = metrics(tech_only)
    tech_strat = stratified_sharpe(tech_only, spy)
    tech_ref = {**tech_m, **tech_strat, "label": "tech_only_reference"}

    # Sweep
    rows = []
    for sig in SIGNALS:
        regime_on = regimes[sig].reindex(common).fillna(False).astype(bool)
        frac_on = float(regime_on.mean())
        for w in WEIGHTS:
            blended = build_gated_book(leader, tech, regime_on, w)
            m = metrics(blended)
            strat = stratified_sharpe(blended, spy)
            row = {
                "signal": sig, "w_tech_when_on": w,
                "regime_on_frac": round(frac_on, 4),
                **m, **strat,
            }
            rows.append(row)

    sweep = pd.DataFrame(rows)
    front = ["signal", "w_tech_when_on", "regime_on_frac", "n_days",
             "cagr_pct", "sharpe", "sortino", "calmar",
             "max_dd_pct", "pf", "wr", "day_conc",
             "sharpe_green", "sharpe_red", "sharpe_flat",
             "n_green", "n_red", "n_flat",
             "regime_gap", "regime_gate_pass"]
    sweep = sweep[front]
    sweep_csv = OUT_DIR / "sweep_results.csv"
    sweep.to_csv(sweep_csv, index=False)

    # Filter: must pass HC #428 R1 gate AND day_conc <= 0.70
    passing = sweep[(sweep["regime_gate_pass"] == True) &
                    (sweep["day_conc"] <= 0.70)].copy()
    any_pass = not passing.empty

    # If any pass, pick best by Calmar then Sharpe
    best = None
    if any_pass:
        passing = passing.sort_values(["calmar", "sharpe"], ascending=False)
        best = passing.iloc[0].to_dict()

    report = {
        "leader_book": str(LEADER_BOOK),
        "tech_book": str(TECH_BOOK),
        "overlap_start": overlap_start,
        "overlap_end": overlap_end,
        "overlap_n_days": int(n_overlap),
        "signals_tested": SIGNALS,
        "weights_tested": WEIGHTS,
        "gates": {
            "hc428_r1_regime_gap_max": 0.50,
            "day_conc_max": 0.70,
        },
        "leader_only_baseline": baseline,
        "tech_only_reference": tech_ref,
        "n_passing_configs": int(passing.shape[0]) if any_pass else 0,
        "best_config": best,
        "all_passing": (passing.to_dict(orient="records") if any_pass else []),
        "full_sweep": sweep.to_dict(orient="records"),
    }
    with open(OUT_DIR / "best_config_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"OUT_DIR: {OUT_DIR}")
    print(f"overlap: {overlap_start} -> {overlap_end}  ({n_overlap} days)")
    print()
    print("LEADER-ONLY BASELINE:")
    for k, v in baseline.items():
        print(f"  {k}: {v}")
    print()
    print("TECH-ONLY REFERENCE:")
    for k, v in tech_ref.items():
        print(f"  {k}: {v}")
    print()
    print("FULL SWEEP:")
    print(sweep.to_string(index=False))
    print()
    if any_pass:
        print(f"PASSING CONFIGS: {len(passing)}")
        print(passing[["signal", "w_tech_when_on", "cagr_pct", "sharpe",
                       "calmar", "max_dd_pct", "regime_gap", "day_conc"]].to_string(index=False))
        print()
        print("BEST CONFIG:")
        for k, v in best.items():
            print(f"  {k}: {v}")
    else:
        print("NO CONFIGS PASS the HC #428 R1 regime gate (|Sharpe_g - Sharpe_r| <= 0.50)")


if __name__ == "__main__":
    main()
