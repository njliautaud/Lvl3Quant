"""
wheel_multi_factor.py — HC #558 R2.

Upgrade the wheel's universe selection from `fund_score >= floor` (binary)
to a *ranked* multi-factor screen that uses ALL the data we have:

  Composite name score = weighted sum of:
    1. Fundamental quality (fund_score)                          [25%]
    2. Sector inflow signal (sector-ETF flow z-score)             [20%]
    3. Sector relative-strength acceleration vs SPY              [15%]
    4. 60-day price momentum of the name itself                  [15%]
    5. Realized-vol stability (rv_60 / rv_252 ratio)             [10%]
    6. Macro health (negative weight when risk_off)              [15%]

Then re-run the Balanced FullWheel tier on the top-N (default 25) ranked
names, real IV + skew + slippage + regime overlay, 2020-2025.

Goal: tighten the universe to names whose sector is being bought AND whose
own price action confirms they're healthy. Should reduce red-month
assignment damage (HC #557 R2) and improve overall Sharpe.

Usage:
    cd /home/jupiter/Lvl3Quant/wheel_strategy_v1
    python3 -m strategy.wheel_multi_factor
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

from strategy.tiers import balanced_tier, _full_wheel, TierSpec
from strategy.tier_runner import _load_inputs, _apply_iv_rank_floor, compute_metrics
from backtest.wheel_engine import run_wheel

ROOT = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1")
CACHE = ROOT / "data" / "cache"
OUT_DIR = ROOT / "results" / "wheel_multi_factor_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2020-01-01"
END = "2025-12-31"
STARTING_CASH = 100_000.0
TOP_N_NAMES = 25

# Universe sector → SPDR mapping
UNIVERSE_TO_ETF = {
    "Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV",
    "Energy": "XLE", "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
    "Industrials": "XLI", "Communication Services": "XLC",
}


def _build_name_scores(prices: pd.DataFrame, universe: pd.DataFrame,
                       fundamentals: pd.DataFrame, etfs: pd.DataFrame,
                       macro: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    """Snapshot score per ticker as of the most recent date.

    We score WITHIN the backtest window — using a snapshot as-of START helps
    avoid lookahead. v1 is fine using as-of START since the universe is
    stable; the wheel engine itself recomputes per-trade gates daily.
    """
    fs = fundamentals.set_index("ticker")

    # Sector ETF stats
    etfs_pivot = etfs.pivot(index="date", columns="ticker", values="close")
    etfs_pivot.index = pd.to_datetime(etfs_pivot.index)
    etfs_dvol = etfs.pivot(index="date", columns="ticker", values="dollar_volume")
    etfs_dvol.index = pd.to_datetime(etfs_dvol.index)

    asof = pd.Timestamp(START)
    if asof not in etfs_pivot.index:
        asof = etfs_pivot.index[etfs_pivot.index.searchsorted(asof) - 1]

    spy = etfs_pivot["SPY"]
    sector_score: Dict[str, dict] = {}
    for sector_name, etf in UNIVERSE_TO_ETF.items():
        if etf not in etfs_pivot.columns:
            continue
        et = etfs_pivot[etf]
        # 60d relative-strength + acceleration
        rs_now = et.loc[:asof].pct_change(60).iloc[-1] - spy.loc[:asof].pct_change(60).iloc[-1]
        if len(et.loc[:asof]) > 90:
            rs_prev = et.loc[:asof].pct_change(60).iloc[-31] - spy.loc[:asof].pct_change(60).iloc[-31]
        else:
            rs_prev = rs_now
        accel = rs_now - rs_prev
        # Flow z
        dv = etfs_dvol[etf].loc[:asof]
        if len(dv) > 20:
            z = (dv.iloc[-1] - dv.iloc[-20:].mean()) / dv.iloc[-20:].std()
            z = float(z) if not pd.isna(z) else 0.0
        else:
            z = 0.0
        sector_score[sector_name] = {"rs": float(rs_now), "accel": float(accel), "flow_z": z}

    # Macro health: 0 if risk_off, 1 if normal
    m = macro.set_index("date").sort_index().loc[:asof]
    r = regime.set_index("date").sort_index().loc[:asof]
    risk_off = bool(r["risk_off"].iloc[-1]) if len(r) else False
    vix = float(m["vix"].iloc[-1]) if "vix" in m.columns and len(m) else 20.0
    macro_health = 1.0 - 0.5 * (1.0 if risk_off else 0.0) - 0.5 * min(1.0, max(0.0, (vix - 20) / 20))

    # Price-derived per-name
    pr = prices.copy()
    pr["date"] = pd.to_datetime(pr["date"])
    pr = pr.sort_values(["ticker", "date"])
    px_pivot = pr.pivot(index="date", columns="ticker", values="close")
    rv60_pivot = pr.pivot(index="date", columns="ticker", values="rv_60")
    rv252_pivot = pr.pivot(index="date", columns="ticker", values="rv_252")

    if asof not in px_pivot.index:
        asof = px_pivot.index[px_pivot.index.searchsorted(asof) - 1]

    px_at = px_pivot.loc[:asof]
    mom_60 = px_at.pct_change(60).iloc[-1]
    rv60 = rv60_pivot.loc[:asof].iloc[-1] if len(rv60_pivot.loc[:asof]) else pd.Series()
    rv252 = rv252_pivot.loc[:asof].iloc[-1] if len(rv252_pivot.loc[:asof]) else pd.Series()
    vol_stability = (rv60 / rv252).replace([np.inf, -np.inf], np.nan)

    rows = []
    for _, u in universe.iterrows():
        tk = u["ticker"]; sector = u["sector"]
        if tk not in fs.index:
            continue
        fund = float(fs.at[tk, "fund_score"])
        sec = sector_score.get(sector, {"rs": 0.0, "accel": 0.0, "flow_z": 0.0})
        # Normalize each component to z-like scale.
        fund_n = (fund - 50) / 20.0   # ~ [-2, 2]
        flow_n = sec["flow_z"]
        accel_n = sec["accel"] * 10.0  # bring to similar magnitude
        mom_n = float(mom_60.get(tk, 0.0)) * 5.0
        # Lower vol_stability (rv60 < rv252) is better → vol coming down
        vs = float(vol_stability.get(tk, 1.0))
        vs_n = (1.0 - vs) * 2.0
        macro_n = 2 * (macro_health - 0.5)

        composite = (0.25 * fund_n + 0.20 * flow_n + 0.15 * accel_n +
                     0.15 * mom_n + 0.10 * vs_n + 0.15 * macro_n)
        rows.append({
            "ticker": tk, "sector": sector,
            "fund_score": fund, "flow_z": flow_n,
            "rs_accel": sec["accel"], "rs_level": sec["rs"],
            "mom_60d": float(mom_60.get(tk, 0.0)), "vol_stability": vs,
            "composite": composite,
        })
    df = pd.DataFrame(rows).sort_values("composite", ascending=False).reset_index(drop=True)
    return df


def filter_multi_factor_balanced(top_n: int = TOP_N_NAMES):
    """Build a closure that picks top_n names by composite score."""
    def _filter(universe, fundamentals, iv_features, prices):
        # We need sector ETFs + macro + regime — load them here.
        etfs = pd.read_parquet(CACHE / "sector_etfs.parquet")
        macro = pd.read_parquet(CACHE / "macro.parquet")
        regime = pd.read_parquet(CACHE / "regime_overlay.parquet")
        scores = _build_name_scores(prices, universe, fundamentals, etfs, macro, regime)
        # Constraint: must have IV history (we already filter prices upstream)
        valid = set(iv_features["ticker"].unique())
        scores = scores[scores["ticker"].isin(valid)]
        chosen = scores.head(top_n)["ticker"].tolist()
        scores.to_parquet(OUT_DIR / "name_scores.parquet")
        return sorted(chosen)
    return _filter


def multi_factor_balanced_tier() -> TierSpec:
    base = balanced_tier()
    return _full_wheel(TierSpec(
        name="Tier2_Balanced_MF",
        target_annual_yield_pct=base.target_annual_yield_pct,
        description=f"{base.description} + multi-factor top-{TOP_N_NAMES} universe (HC #558 R2)",
        wheel_cfg=base.wheel_cfg,
        universe_filter=filter_multi_factor_balanced(TOP_N_NAMES),
        iv_rank_floor=base.iv_rank_floor,
        regime_sensitivity=base.regime_sensitivity,
        capital_share_default=base.capital_share_default,
    ))


def main():
    import strategy.tier_runner as TR
    from backtest import wheel_engine as WE
    WE.USE_SKEW = True
    WE.USE_SLIPPAGE = True
    WE.SLIPPAGE_FRAC = 0.025
    WE.SLIPPAGE_MIN_TICKS = 0.03

    data = _load_inputs(modeled=True, smoke=False, real_iv=True)
    spec = multi_factor_balanced_tier()
    tickers = spec.universe_filter(data["universe"], data["fundamentals"],
                                   data["iv"], data["prices"])
    print(f"[MF] composite-ranked top-{TOP_N_NAMES}: {tickers}")

    px = data["prices"][data["prices"]["ticker"].isin(tickers)].copy()
    iv = data["iv"][data["iv"]["ticker"].isin(tickers)].copy()
    iv = _apply_iv_rank_floor(iv, spec.iv_rank_floor)

    res = run_wheel(
        cfg=spec.wheel_cfg, prices=px, iv=iv, macro=data["macro"],
        fundamentals=data["fundamentals"], universe=data["universe"],
        starting_cash=STARTING_CASH, start=START, end=END, verbose=False,
    )
    metrics = compute_metrics(res, STARTING_CASH)

    # Persist
    if "ledger" in res and res["ledger"] is not None:
        res["ledger"].to_parquet(OUT_DIR / "ledger.parquet")
    if "equity" in res and res["equity"] is not None:
        res["equity"].to_parquet(OUT_DIR / "equity.parquet")
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                   for k, v in metrics.items()}, f, indent=2, default=str)

    report = f"""# Wheel Multi-Factor Universe — HC #558 R2

Window: {START} → {END}, $100k. Real IV + skew + slippage + regime overlay.
Universe: composite-ranked top-{TOP_N_NAMES} from 70-name pool.

Composite weights:
  25% fund_score | 20% sector flow z | 15% sector RS acceleration |
  15% name 60d momentum | 10% vol stability | 15% macro health

## Selected universe
{', '.join(tickers)}

## Headline metrics

| Metric | Multi-Factor Balanced | v7 broad Balanced (HC #557 baseline) |
|---|---|---|
| Realized CAGR | {metrics.get('realized_cagr', 0):.1%} | 10.6% |
| Realized Sharpe | {metrics.get('realized_sharpe', 0):.2f} | 1.02 |
| Realized Sortino | {metrics.get('realized_sortino', 0):.2f} | 0.41 |
| Realized MaxDD | {metrics.get('realized_max_dd', 0):.1%} | -11.9% |
| WR | {metrics.get('wr', 0):.1%} | n/a |
| Assignment rate | {metrics.get('assignment_rate', 0):.1%} | n/a |
| n_trades | {metrics.get('n_trades', 0):,} | n/a |

vs SPY 1.5x margin (Sharpe 0.70, MaxDD -47%): multi-factor wheel wins if Realized Sharpe > 0.70.
"""
    (OUT_DIR / "report.md").write_text(report)
    print(f"\n[MF] Report → {OUT_DIR / 'report.md'}")
    print(f"[MF] Realized Sharpe: {metrics.get('realized_sharpe', 0):.2f}  CAGR: {metrics.get('realized_cagr', 0):.2%}  DD: {metrics.get('realized_max_dd', 0):.1%}")


if __name__ == "__main__":
    main()
