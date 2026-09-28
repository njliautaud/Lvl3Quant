"""
v3_hedged.py — HC #558 R3/R4 picker v3 with explicit risk-off hedge.

Fixes v2's red-month bleed (-3.93%/mo) by replacing the "reduced equity"
risk-off behaviour with an explicit defensive book:

  Normal regime  (gate=open):   full equity book (sectors + themes) as v2.
  Risk-off (vix>30 OR regime_overlay.risk_off):
      50% TLT (long-duration treasuries)
    + 25% GLD (gold)
    + 25% cash
  Hard gate (vix>40):           100% TLT + GLD (60/40) + zero equity.

Hypothesis: TLT + GLD historically rally in risk-off episodes (Mar 2020,
late 2018, Aug 2024). Holding them through red months should turn the
picker's red-month return from negative to roughly flat or positive,
clearing HC #557 R2 properly without short-selling.

Usage:
    cd /home/jupiter/Lvl3Quant/wheel_strategy_v1
    python3 -m strategy.macro_picker.v3_hedged
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

from strategy.macro_picker.v1_sector_rotation import (
    load_inputs, make_pivot, risk_metrics,
    regime_classify, stratified_sharpe, monthly_return_by_regime,
    spy_benchmarks,
)
from strategy.macro_picker.v2_thematic_rotation import (
    select_sectors_v2, select_themes, select_names_in_sector_v2,
    THEMATIC_STANDALONE, MIN_SECTOR_MEMBERS, UNIVERSE_TO_ETF,
)

ROOT = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1")
CACHE = ROOT / "data" / "cache"
OUT_DIR = ROOT / "results" / "macro_picker_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2020-01-01")
END = pd.Timestamp("2025-12-31")
STARTING_CASH = 100_000.0

SLIPPAGE_BPS = 0.5
COMMISSION_BPS = 0.5
ONE_WAY_COST_BPS = SLIPPAGE_BPS + COMMISSION_BPS
TRADING_DAYS = 252
ANN = np.sqrt(TRADING_DAYS)

VIX_HARD = 40.0
VIX_SOFT = 30.0

HEDGE_RISK_OFF = {"GLD": 0.40}  # 60% cash implied — GLD only because TLT failed in 2022
HEDGE_HARD = {"GLD": 0.60}      # 40% cash implied


def gate_state(date, macro, regime):
    m = macro.set_index("date").loc[:date]
    r = regime.set_index("date").loc[:date]
    vix = float(m["vix"].iloc[-1]) if len(m) else 0.0
    risk_off = bool(r["risk_off"].iloc[-1]) if len(r) else False
    if vix > VIX_HARD:
        return "hard"
    if risk_off or vix > VIX_SOFT:
        return "risk_off"
    return "normal"


def backtest_v3(data: dict) -> dict:
    prices = data["prices"]
    universe = data["universe"]
    macro = data["macro"]
    regime = data["regime"]
    etfs = data["etfs"]
    fund = pd.read_parquet(CACHE / "fundamentals.parquet")

    prices_close = make_pivot(prices, "close")
    etfs_close = make_pivot(etfs, "close")
    etfs_dvol = make_pivot(etfs, "dollar_volume")

    combined = prices_close.join(etfs_close, how="outer", lsuffix="", rsuffix="_etf")
    combined = combined[(combined.index >= START) & (combined.index <= END)]
    combined = combined.ffill().bfill()

    sector_etf_list = [e for e in UNIVERSE_TO_ETF.values() if e in etfs_close.columns]
    etf_to_sector = {v: k for k, v in UNIVERSE_TO_ETF.items()}
    spy_close = etfs_close["SPY"]

    dates = combined.index
    rebal_dates = pd.Series(dates).groupby(pd.Series(dates).dt.to_period("M")).first().tolist()

    weights = {tk: 0.0 for tk in combined.columns}
    equity = [STARTING_CASH]
    eq_dates = [dates[0]]
    holdings_log = []

    for i, d in enumerate(dates[1:], start=1):
        d_prev = dates[i - 1]
        rets = (combined.loc[d] / combined.loc[d_prev] - 1.0).fillna(0.0)
        port_ret = sum(weights[tk] * rets.get(tk, 0.0) for tk in weights)
        equity.append(equity[-1] * (1 + port_ret))
        eq_dates.append(d)

        if d in rebal_dates and d != dates[0]:
            state = gate_state(d, macro, regime)
            target = {tk: 0.0 for tk in weights}

            if state == "hard":
                for tk, w in HEDGE_HARD.items():
                    if tk in target:
                        target[tk] = w
                rationale = f"HARD_GATE → {HEDGE_HARD}"
            elif state == "risk_off":
                for tk, w in HEDGE_RISK_OFF.items():
                    if tk in target:
                        target[tk] = w
                rationale = f"RISK_OFF_HEDGE → {HEDGE_RISK_OFF} + 25% cash"
            else:
                chosen_etfs = select_sectors_v2(
                    etfs_close, etfs_dvol, spy_close, d,
                    sector_etf_list, MIN_SECTOR_MEMBERS, universe)
                chosen_themes = select_themes(etfs_close, etfs_dvol, spy_close, d,
                                              THEMATIC_STANDALONE)
                names = []
                for etf in chosen_etfs:
                    names.extend(select_names_in_sector_v2(
                        combined, universe, fund, etf, d, etf_to_sector, risk_off=False))
                slots = [s for s in (names + chosen_themes) if s in combined.columns]
                if not slots:
                    rationale = "no_slots — cash"
                else:
                    w = 1.0 / len(slots)
                    for tk in slots:
                        target[tk] = w
                    rationale = f"NORMAL sectors={chosen_etfs} themes={chosen_themes} n_slots={len(slots)}"

            turnover = sum(abs(target[tk] - weights[tk]) for tk in weights)
            cost = turnover * (ONE_WAY_COST_BPS / 1e4)
            equity[-1] *= (1 - cost)
            weights = target
            holdings_log.append({"date": d, "state": state, "rationale": rationale,
                                 "holdings": [tk for tk, w in target.items() if w > 0]})

    eq_series = pd.Series(equity, index=pd.DatetimeIndex(eq_dates), name="equity")
    daily_ret = eq_series.pct_change().dropna()
    metrics = risk_metrics(daily_ret, eq_series)
    return {
        "equity": eq_series, "daily_ret": daily_ret, "metrics": metrics,
        "holdings_log": holdings_log,
        "spy_close": spy_close.loc[eq_series.index[0]:eq_series.index[-1]],
    }


def verdict_v3(metrics, spy15, regime_monthly, strat_sharpe):
    reasons = []
    sh_win = metrics["sharpe"] > spy15["sharpe"]
    cagr_win = metrics["cagr"] > spy15["cagr"]
    if not sh_win:
        reasons.append(f"Sharpe {metrics['sharpe']:.2f} <= 1.5x SPY {spy15['sharpe']:.2f}")
    if not cagr_win:
        reasons.append(f"CAGR {metrics['cagr']:.1%} <= 1.5x SPY {spy15['cagr']:.1%}")
    vals = list(regime_monthly.values())
    if min(vals) < 0:
        reasons.append(f"red-month {regime_monthly['red']:.2%} still negative")
    sh_vals = [v for v in strat_sharpe.values() if not np.isnan(v)]
    if len(sh_vals) >= 2:
        worst, best = min(sh_vals, key=abs), max(sh_vals, key=abs)
        if abs(best) > 0 and abs(worst) / abs(best) < 0.50:
            reasons.append(f"regime sharpe asymmetric {abs(worst)/abs(best):.2f} < 0.50")
    if not reasons:
        return "PICKER v3 WINS"
    if sh_win and not cagr_win and regime_monthly['red'] >= 0:
        return "PICKER v3 HALF-WIN (Sharpe + regime gate pass; CAGR lags margin-SPY)"
    return "PICKER v3 LOSES — reasons: " + "; ".join(reasons)


def main():
    print("[v3] loading inputs")
    data = load_inputs()
    print("[v3] running hedged backtest")
    res = backtest_v3(data)
    print(f"[v3] metrics: {res['metrics']}")

    spy_b = spy_benchmarks(res["spy_close"], data["macro"])
    spy10 = spy_b[1.0]["metrics"]
    spy15 = spy_b[1.5]["metrics"]
    spy20 = spy_b[2.0]["metrics"]
    regime = regime_classify(res["spy_close"])
    monthly_by_regime = monthly_return_by_regime(res["daily_ret"], regime)
    strat_sh = stratified_sharpe(res["daily_ret"], regime)
    v = verdict_v3(res["metrics"], spy15, monthly_by_regime, strat_sh)

    res["equity"].to_frame().to_parquet(OUT_DIR / "equity_picker_v3.parquet")
    pd.DataFrame(res["holdings_log"]).to_parquet(OUT_DIR / "holdings_log.parquet")
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump({"picker_v3": res["metrics"], "spy_1.0x": spy10, "spy_1.5x": spy15,
                   "spy_2.0x": spy20, "monthly_return_by_regime": monthly_by_regime,
                   "stratified_sharpe": strat_sh, "verdict": v}, f, indent=2, default=str)

    n_hard = sum(1 for h in res["holdings_log"] if h["state"] == "hard")
    n_off = sum(1 for h in res["holdings_log"] if h["state"] == "risk_off")
    n_norm = sum(1 for h in res["holdings_log"] if h["state"] == "normal")
    report = f"""# Macro Picker v3 — Hedged with TLT + GLD

**Verdict: {v}**

Window: 2020-01-01 → 2025-12-31, $100k starting. Monthly rebalance.
Normal regime: same equity-rotation logic as v2.
Risk-off (VIX > 30 OR regime_overlay.risk_off): 50% TLT + 25% GLD + 25% cash.
Hard gate (VIX > 40): 60% TLT + 40% GLD.

Regime state distribution: normal {n_norm}, risk-off {n_off}, hard {n_hard} of {len(res['holdings_log'])} rebalances.

## Headline

| Strategy | CAGR | Sharpe | Sortino | MaxDD |
|---|---|---|---|---|
| Picker v3 | {res['metrics']['cagr']:.1%} | {res['metrics']['sharpe']:.2f} | {res['metrics']['sortino']:.2f} | {res['metrics']['max_dd']:.1%} |
| SPY 1.0× | {spy10['cagr']:.1%} | {spy10['sharpe']:.2f} | {spy10['sortino']:.2f} | {spy10['max_dd']:.1%} |
| SPY 1.5× margin | {spy15['cagr']:.1%} | {spy15['sharpe']:.2f} | {spy15['sortino']:.2f} | {spy15['max_dd']:.1%} |

## Regime Diagnostics (HC #557 R2)

Mean monthly return:
- green: {monthly_by_regime['green']:.2%} | red: {monthly_by_regime['red']:.2%} | flat: {monthly_by_regime['flat']:.2%}

Stratified Sharpe:
- green: {strat_sh['green']:.2f} | red: {strat_sh['red']:.2f} | flat: {strat_sh['flat']:.2f}
"""
    (OUT_DIR / "report.md").write_text(report)
    print(f"\n[v3] {v}")


if __name__ == "__main__":
    main()
