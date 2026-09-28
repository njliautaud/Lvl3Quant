"""
v2_thematic_rotation.py — HC #558 R3/R4 macro stock-picker v2.

Fixes v1's three biggest losses:

  (1) Macro gate too trigger-happy.
      v1 went 100% cash on 33/71 rebalances. v2 only stands down on:
        - VIX > 40 (was 35), AND
        - regime_overlay.risk_off=True AND vix > 30
      Otherwise we trade at reduced size (50% if risk_off, 100% else).

  (2) Sector signal looked backward.
      v1 used 60d relative-strength level. v2 uses momentum acceleration:
        rs_60d - rs_60d_30days_ago  (positive = accelerating, getting stronger)
      This catches rotation EARLIER instead of after it has fully run.

  (3) Picked sectors with no universe members.
      v2 only ranks sectors that map to at least 3 names in our universe,
      AND adds standalone thematic slots (SMH/IGV/SOXX/XBI/ARKK) that can
      be allocated even without member names — we just hold the ETF.

  (4) Bonus: add factor tilt — within each chosen sector/theme, score names by
      a composite of:
        - 60d momentum (40%)
        - quality (fund_score, 30%)
        - low-vol bias for risk-off, high-vol bias for risk-on (30%)

Targets:
  Picker v2 must clear HC #558 R5: Sharpe > margin-SPY 1.5x AND CAGR > margin-SPY 1.5x.
  Picker v2 must clear HC #557 R2: positive monthly return in all regimes,
  worst/best regime Sharpe ratio >= 0.50.

Usage:
    cd /home/jupiter/Lvl3Quant/wheel_strategy_v1
    python3 -m strategy.macro_picker.v2_thematic_rotation
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
    load_inputs, make_pivot, zscore, risk_metrics,
    regime_classify, stratified_sharpe, monthly_return_by_regime,
    spy_benchmarks, UNIVERSE_TO_ETF,
)

ROOT = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1")
CACHE = ROOT / "data" / "cache"
OUT_DIR = ROOT / "results" / "macro_picker_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2020-01-01")
END = pd.Timestamp("2025-12-31")
STARTING_CASH = 100_000.0

SLIPPAGE_BPS = 0.5
COMMISSION_BPS = 0.5
ONE_WAY_COST_BPS = SLIPPAGE_BPS + COMMISSION_BPS
TRADING_DAYS = 252
ANN = np.sqrt(TRADING_DAYS)

# Thematic ETFs that can be held standalone (don't need universe members)
THEMATIC_STANDALONE = ["SMH", "SOXX", "IGV", "XBI", "ARKK"]
# Cap total slots = sectors_chosen × names_per_sector + thematic_slots
TOP_N_SECTORS = 3
NAMES_PER_SECTOR = 5
TOP_N_THEMES = 2
MIN_SECTOR_MEMBERS = 3

LOOKBACK_RS = 60
LOOKBACK_RS_DELTA = 30  # acceleration window
LOOKBACK_NAME_MOM = 60
LOOKBACK_FLOW = 20
LOOKBACK_VOL = 20

VIX_HARD_GATE = 40.0
VIX_RISK_OFF_GATE = 30.0


# -------------------- Signals --------------------

def sector_universe_counts(universe: pd.DataFrame) -> Dict[str, int]:
    """How many universe names per SPDR sector?"""
    inv = {v: k for k, v in UNIVERSE_TO_ETF.items()}
    counts = {}
    for etf, sector in inv.items():
        n = (universe["sector"] == sector).sum()
        counts[etf] = int(n)
    return counts


def select_sectors_v2(etfs_close: pd.DataFrame, etfs_dvol: pd.DataFrame,
                      spy_close: pd.Series, date: pd.Timestamp,
                      sector_etf_list: List[str], min_members: int,
                      universe: pd.DataFrame) -> List[str]:
    """Top-N sectors by acceleration of relative-strength vs SPY."""
    if date not in etfs_close.index:
        try:
            date = etfs_close.index[etfs_close.index.searchsorted(date) - 1]
        except Exception:
            return []
    if etfs_close.index.get_loc(date) < LOOKBACK_RS + LOOKBACK_RS_DELTA:
        return []

    counts = sector_universe_counts(universe)
    score = {}
    for tk in sector_etf_list:
        if tk not in etfs_close.columns:
            continue
        if counts.get(tk, 0) < min_members:
            continue
        ser_etf = etfs_close[tk].loc[:date]
        ser_spy = spy_close.loc[:date]
        # current 60d RS
        rs_now = ser_etf.pct_change(LOOKBACK_RS).iloc[-1] - ser_spy.pct_change(LOOKBACK_RS).iloc[-1]
        # 60d RS as-of (LOOKBACK_RS_DELTA) days ago
        if len(ser_etf) <= LOOKBACK_RS + LOOKBACK_RS_DELTA:
            continue
        rs_prev_etf = ser_etf.pct_change(LOOKBACK_RS).iloc[-1 - LOOKBACK_RS_DELTA]
        rs_prev_spy = ser_spy.pct_change(LOOKBACK_RS).iloc[-1 - LOOKBACK_RS_DELTA]
        rs_prev = rs_prev_etf - rs_prev_spy
        if pd.isna(rs_now) or pd.isna(rs_prev):
            continue
        accel = rs_now - rs_prev  # positive = accelerating relative strength
        # Flow z (positive bonus only)
        dvol = etfs_dvol[tk].loc[:date]
        z = zscore(dvol, LOOKBACK_FLOW).iloc[-1] if len(dvol) > LOOKBACK_FLOW else 0.0
        z = float(z) if not pd.isna(z) else 0.0
        flow_bonus = 0.10 * max(0.0, z)
        # Combined: accel is primary, level is secondary tiebreaker
        score[tk] = float(accel) + 0.30 * float(rs_now) + flow_bonus

    ordered = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    return [tk for tk, _ in ordered[:TOP_N_SECTORS]]


def select_themes(etfs_close: pd.DataFrame, etfs_dvol: pd.DataFrame,
                  spy_close: pd.Series, date: pd.Timestamp,
                  themes: List[str]) -> List[str]:
    """Top-N thematic ETFs by RS acceleration vs SPY."""
    if etfs_close.index.get_loc(date) < LOOKBACK_RS + LOOKBACK_RS_DELTA:
        return []
    score = {}
    for tk in themes:
        if tk not in etfs_close.columns:
            continue
        ser = etfs_close[tk].loc[:date]
        if len(ser) <= LOOKBACK_RS + LOOKBACK_RS_DELTA:
            continue
        rs_now = ser.pct_change(LOOKBACK_RS).iloc[-1] - spy_close.loc[:date].pct_change(LOOKBACK_RS).iloc[-1]
        rs_prev = ser.pct_change(LOOKBACK_RS).iloc[-1 - LOOKBACK_RS_DELTA] - \
                  spy_close.loc[:date].pct_change(LOOKBACK_RS).iloc[-1 - LOOKBACK_RS_DELTA]
        if pd.isna(rs_now) or pd.isna(rs_prev):
            continue
        accel = rs_now - rs_prev
        # Theme only included if accelerating AND currently outperforming
        if accel <= 0 or rs_now <= 0:
            continue
        score[tk] = float(accel) + 0.30 * float(rs_now)
    ordered = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    return [tk for tk, _ in ordered[:TOP_N_THEMES]]


def select_names_in_sector_v2(prices_close: pd.DataFrame, universe: pd.DataFrame,
                              fundamentals: pd.DataFrame, etf_ticker: str,
                              date: pd.Timestamp, etf_to_sector: Dict[str, str],
                              risk_off: bool) -> List[str]:
    """Composite-scored top names in the sector."""
    sector_name = etf_to_sector.get(etf_ticker)
    if sector_name is None:
        return []
    sector_tickers = universe[universe["sector"] == sector_name]["ticker"].tolist()
    sector_tickers = [t for t in sector_tickers if t in prices_close.columns]
    if not sector_tickers:
        return []
    if prices_close.index.get_loc(date) < LOOKBACK_NAME_MOM:
        return []
    px = prices_close.loc[:date, sector_tickers]
    mom = px.pct_change(LOOKBACK_NAME_MOM).iloc[-1].dropna()
    # Realized vol over LOOKBACK_VOL
    vol = px.pct_change().iloc[-LOOKBACK_VOL:].std()
    fs = fundamentals.set_index("ticker")
    score = {}
    for tk in mom.index:
        m = float(mom[tk])
        v = float(vol.get(tk, np.nan))
        f = float(fs.at[tk, "fund_score"]) if tk in fs.index else 50.0
        if np.isnan(v) or v <= 0:
            v_score = 0.0
        else:
            # In risk-off, prefer low vol; in risk-on, prefer high vol
            v_score = -v if risk_off else v
        # Normalize and combine: 40% mom, 30% fund, 30% vol-bias
        score[tk] = 0.40 * m + 0.30 * (f / 100.0) + 0.30 * v_score
    ordered = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    return [tk for tk, _ in ordered[:NAMES_PER_SECTOR]]


def macro_gate_v2(date: pd.Timestamp, macro_df: pd.DataFrame,
                  regime_df: pd.DataFrame) -> float:
    """Return position-size multiplier ∈ {0, 0.5, 1.0}."""
    m = macro_df.set_index("date")
    r = regime_df.set_index("date")
    m_row = m.loc[:date].iloc[-1] if len(m.loc[:date]) else None
    r_row = r.loc[:date].iloc[-1] if len(r.loc[:date]) else None
    vix = float(m_row.get("vix", 0.0)) if m_row is not None else 0.0
    risk_off = bool(r_row.get("risk_off", False)) if r_row is not None else False
    if vix > VIX_HARD_GATE:
        return 0.0
    if risk_off and vix > VIX_RISK_OFF_GATE:
        return 0.0
    if risk_off:
        return 0.5  # reduced size in risk-off but not paralyzed
    return 1.0


# -------------------- Backtest --------------------

def backtest_v2(data: dict) -> dict:
    prices = data["prices"]
    universe = data["universe"]
    macro = data["macro"]
    regime = data["regime"]
    etfs = data["etfs"]
    fund = pd.read_parquet(CACHE / "fundamentals.parquet")

    prices_close = make_pivot(prices, "close")
    etfs_close = make_pivot(etfs, "close")
    etfs_dvol = make_pivot(etfs, "dollar_volume")

    # Add ETFs to price universe so we can hold them
    etf_tickers = list(etfs_close.columns)
    combined_close = prices_close.join(etfs_close[etf_tickers], how="outer", lsuffix="", rsuffix="_etf")
    # When joining, etfs that overlap with universe (none should) won't collide

    combined_close = combined_close[(combined_close.index >= START) & (combined_close.index <= END)]
    combined_close = combined_close.ffill().bfill()

    sector_etf_list = [e for e in UNIVERSE_TO_ETF.values() if e in etfs_close.columns]
    etf_to_sector = {v: k for k, v in UNIVERSE_TO_ETF.items()}
    spy_close = etfs_close["SPY"]

    dates = combined_close.index
    rebal_dates = pd.Series(dates).groupby(pd.Series(dates).dt.to_period("M")).first().tolist()

    weights = {tk: 0.0 for tk in combined_close.columns}
    equity = [STARTING_CASH]
    eq_dates = [dates[0]]
    holdings_log = []

    for i, d in enumerate(dates[1:], start=1):
        d_prev = dates[i - 1]
        rets = (combined_close.loc[d] / combined_close.loc[d_prev] - 1.0).fillna(0.0)
        port_ret = sum(weights[tk] * rets.get(tk, 0.0) for tk in weights)
        equity.append(equity[-1] * (1 + port_ret))
        eq_dates.append(d)

        if d in rebal_dates and d != dates[0]:
            size_mult = macro_gate_v2(d, macro, regime)
            risk_off_now = size_mult < 1.0

            target = {tk: 0.0 for tk in weights}
            if size_mult == 0.0:
                rationale = "macro_gate_closed_full"
            else:
                chosen_etfs = select_sectors_v2(
                    etfs_close, etfs_dvol, spy_close, d,
                    sector_etf_list, MIN_SECTOR_MEMBERS, universe)
                chosen_themes = select_themes(etfs_close, etfs_dvol, spy_close, d,
                                              THEMATIC_STANDALONE)
                chosen_names: List[str] = []
                for etf in chosen_etfs:
                    chosen_names.extend(select_names_in_sector_v2(
                        combined_close, universe, fund, etf, d, etf_to_sector,
                        risk_off=risk_off_now))
                slots = chosen_names + chosen_themes
                slots = [s for s in slots if s in combined_close.columns]
                if not slots:
                    rationale = "no_slots_selected"
                else:
                    base_w = size_mult / len(slots)
                    for tk in slots:
                        target[tk] = base_w
                    rationale = f"size={size_mult:.1f} sectors={chosen_etfs} themes={chosen_themes} n_slots={len(slots)}"

            turnover = sum(abs(target[tk] - weights[tk]) for tk in weights)
            cost = turnover * (ONE_WAY_COST_BPS / 1e4)
            equity[-1] *= (1 - cost)
            weights = target
            holdings_log.append({
                "date": d,
                "rationale": rationale,
                "holdings": [tk for tk, w in target.items() if w > 0],
            })

    eq_series = pd.Series(equity, index=pd.DatetimeIndex(eq_dates), name="equity")
    daily_ret = eq_series.pct_change().dropna()
    metrics = risk_metrics(daily_ret, eq_series)
    return {
        "equity": eq_series, "daily_ret": daily_ret, "metrics": metrics,
        "holdings_log": holdings_log,
        "spy_close": spy_close.loc[eq_series.index[0]:eq_series.index[-1]],
    }


def verdict_v2(metrics, spy15, regime_monthly, strat_sharpe):
    reasons = []
    sh_win = metrics["sharpe"] > spy15["sharpe"]
    cagr_win = metrics["cagr"] > spy15["cagr"]
    if not sh_win:
        reasons.append(f"Sharpe {metrics['sharpe']:.2f} <= margin-SPY 1.5x {spy15['sharpe']:.2f}")
    if not cagr_win:
        reasons.append(f"CAGR {metrics['cagr']:.1%} <= margin-SPY 1.5x {spy15['cagr']:.1%}")
    vals = [v for v in regime_monthly.values()]
    if min(vals) < 0:
        reasons.append(f"monthly return negative in some regime: { {k: f'{v:.2%}' for k,v in regime_monthly.items()} }")
    sh_vals = [v for v in strat_sharpe.values() if not np.isnan(v)]
    if len(sh_vals) >= 2:
        worst = min(sh_vals, key=abs); best = max(sh_vals, key=abs)
        if abs(best) > 0 and abs(worst) / abs(best) < 0.50:
            reasons.append(f"regime sharpe asymmetric worst/best={abs(worst)/abs(best):.2f} < 0.50")
    if not reasons:
        return "PICKER v2 WINS"
    if sh_win and not cagr_win and not any("regime" in r or "monthly" in r for r in reasons):
        return "PICKER v2 HALF-WIN (Sharpe beats margin-SPY but CAGR loses; acceptable per HC #557)"
    return "PICKER v2 LOSES — reasons: " + "; ".join(reasons)


def main():
    print("[picker v2] loading inputs")
    data = load_inputs()
    print("[picker v2] running backtest")
    res = backtest_v2(data)
    print(f"[picker v2] metrics: {res['metrics']}")

    spy_b = spy_benchmarks(res["spy_close"], data["macro"])
    spy10 = spy_b[1.0]["metrics"]
    spy15 = spy_b[1.5]["metrics"]
    spy20 = spy_b[2.0]["metrics"]

    regime = regime_classify(res["spy_close"])
    monthly_by_regime = monthly_return_by_regime(res["daily_ret"], regime)
    strat_sh = stratified_sharpe(res["daily_ret"], regime)
    v = verdict_v2(res["metrics"], spy15, monthly_by_regime, strat_sh)

    res["equity"].to_frame().to_parquet(OUT_DIR / "equity_picker_v2.parquet")
    pd.DataFrame(res["holdings_log"]).to_parquet(OUT_DIR / "holdings_log.parquet")
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump({
            "picker_v2": res["metrics"],
            "spy_1.0x": spy10, "spy_1.5x": spy15, "spy_2.0x": spy20,
            "monthly_return_by_regime": monthly_by_regime,
            "stratified_sharpe": strat_sh,
            "verdict": v,
        }, f, indent=2, default=str)

    n_stand = sum(1 for h in res["holdings_log"] if "gate_closed" in h["rationale"] or "no_slots" in h["rationale"])
    report = f"""# Macro Picker v2 — Thematic Rotation + Acceleration

**Verdict: {v}**

Window: 2020-01-01 → 2025-12-31, $100k starting. Monthly rebalance. Long-only.
Sectors selected by acceleration of 60d relative-strength vs SPY (forward-looking, catches rotation earlier).
Standalone thematic slots: SMH, SOXX, IGV, XBI, ARKK (top 2 by same acceleration signal).
Macro gate: 100% cash only if VIX > 40 OR (risk_off AND VIX > 30). Reduced 50% in risk_off otherwise.

Stand-downs: {n_stand} / {len(res['holdings_log'])} rebalances.

## Headline

| Strategy | CAGR | Sharpe | Sortino | MaxDD |
|---|---|---|---|---|
| Picker v2 | {res['metrics']['cagr']:.1%} | {res['metrics']['sharpe']:.2f} | {res['metrics']['sortino']:.2f} | {res['metrics']['max_dd']:.1%} |
| SPY 1.0× | {spy10['cagr']:.1%} | {spy10['sharpe']:.2f} | {spy10['sortino']:.2f} | {spy10['max_dd']:.1%} |
| SPY 1.5× margin | {spy15['cagr']:.1%} | {spy15['sharpe']:.2f} | {spy15['sortino']:.2f} | {spy15['max_dd']:.1%} |
| SPY 2.0× margin | {spy20['cagr']:.1%} | {spy20['sharpe']:.2f} | {spy20['sortino']:.2f} | {spy20['max_dd']:.1%} |

## Regime Diagnostics (HC #557 R2)

Mean monthly return:
- green: {monthly_by_regime['green']:.2%} | red: {monthly_by_regime['red']:.2%} | flat: {monthly_by_regime['flat']:.2%}

Stratified Sharpe:
- green: {strat_sh['green']:.2f} | red: {strat_sh['red']:.2f} | flat: {strat_sh['flat']:.2f}
"""
    (OUT_DIR / "report.md").write_text(report)
    print(f"\n[picker v2] {v}")


if __name__ == "__main__":
    main()
