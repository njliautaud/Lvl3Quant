"""
Walk-forward harness (HC #561 R3 + HC #559 R1).

Generic harness consumed by GA fits, ranking-model training, and wheel sweeps.
Produces per-fold metrics + verdict per HC #561 R4 acceptance gates.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# acceptance gates (HC #561 R4)
# Floor    : Calmar ≥ 1.0
# Target   : CAGR ≥ 0.18, Calmar ≥ 1.5, Sharpe ≥ 1.5, MaxDD ≤ 0.15
# Stretch  : CAGR ≥ 0.25, Calmar ≥ 2.0
# ---------------------------------------------------------------------------
TRADING_DAYS = 252


def _annualize_sharpe(rets: pd.Series) -> float:
    rets = rets.dropna()
    if len(rets) < 2 or rets.std() == 0:
        return float("nan")
    return float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS))


def _annualize_sortino(rets: pd.Series) -> float:
    rets = rets.dropna()
    if len(rets) < 2:
        return float("nan")
    downside = rets[rets < 0]
    if len(downside) < 1 or downside.std() == 0:
        return float("nan")
    return float(rets.mean() / downside.std() * np.sqrt(TRADING_DAYS))


def _cagr(rets: pd.Series) -> float:
    rets = rets.dropna()
    if len(rets) < 2:
        return float("nan")
    eq = (1.0 + rets).cumprod()
    n_years = len(rets) / TRADING_DAYS
    if n_years <= 0:
        return float("nan")
    return float(eq.iloc[-1] ** (1.0 / n_years) - 1.0)


def _max_dd(rets: pd.Series) -> float:
    rets = rets.dropna()
    if len(rets) < 2:
        return float("nan")
    eq = (1.0 + rets).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return float(dd.min())  # negative


def _calmar(rets: pd.Series) -> float:
    cagr = _cagr(rets)
    mdd = _max_dd(rets)
    if not np.isfinite(cagr) or not np.isfinite(mdd) or mdd == 0:
        return float("nan")
    return float(cagr / abs(mdd))


def _profit_factor(rets: pd.Series) -> float:
    rets = rets.dropna()
    pos = rets[rets > 0].sum()
    neg = -rets[rets < 0].sum()
    if neg == 0:
        return float("nan")
    return float(pos / neg)


def _win_rate(rets: pd.Series) -> float:
    rets = rets.dropna()
    if len(rets) < 1:
        return float("nan")
    return float((rets > 0).mean())


def _spy_levered(spy: pd.Series, lev: float, fund_rate: pd.Series, spread_bps: int) -> pd.Series:
    """SPY at leverage `lev` with margin financing cost."""
    daily_rate = ((fund_rate / 100.0 + spread_bps / 10000.0) / TRADING_DAYS).reindex(spy.index).ffill()
    excess_lev = max(lev - 1.0, 0.0)
    return lev * spy - excess_lev * daily_rate


def _metrics(rets: pd.Series) -> Dict[str, float]:
    return {
        "sharpe": _annualize_sharpe(rets),
        "sortino": _annualize_sortino(rets),
        "cagr": _cagr(rets),
        "max_dd": _max_dd(rets),
        "calmar": _calmar(rets),
        "pf": _profit_factor(rets),
        "wr": _win_rate(rets),
        "n_trades": int(rets.dropna().shape[0]),
    }


def _verdict_for_metrics(m: Dict[str, float]) -> str:
    cagr = m.get("cagr", float("nan"))
    calmar = m.get("calmar", float("nan"))
    sharpe = m.get("sharpe", float("nan"))
    mdd = m.get("max_dd", float("nan"))
    if not np.isfinite(calmar) or calmar < 1.0:
        return "FAILS CALMAR FLOOR"
    if (np.isfinite(cagr) and cagr >= 0.25 and np.isfinite(calmar) and calmar >= 2.0):
        return "STRETCH MET"
    if (
        np.isfinite(cagr) and cagr >= 0.18
        and np.isfinite(calmar) and calmar >= 1.5
        and np.isfinite(sharpe) and sharpe >= 1.5
        and np.isfinite(mdd) and mdd >= -0.15
    ):
        return "TARGET MET"
    return "VIABLE BUT UNDER-TARGET"


@dataclass
class WalkForwardResult:
    per_fold: List[Dict] = field(default_factory=list)
    summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
    verdict: str = ""

    def verdict_for(self) -> str:
        return self.verdict


def _summarize(per_fold: List[Dict]) -> Dict[str, Dict[str, float]]:
    if not per_fold:
        return {}
    keys = [k for k in per_fold[0].keys() if isinstance(per_fold[0][k], (int, float))]
    out = {}
    for k in keys:
        vals = pd.Series([f.get(k, np.nan) for f in per_fold], dtype=float).dropna()
        if len(vals) == 0:
            out[k] = {"median": float("nan"), "p25": float("nan"), "p75": float("nan")}
        else:
            out[k] = {
                "median": float(vals.median()),
                "p25": float(vals.quantile(0.25)),
                "p75": float(vals.quantile(0.75)),
            }
    return out


def _summary_verdict(summary: Dict[str, Dict[str, float]]) -> str:
    """Verdict on the SUMMARY medians (HC #561 R4 + HC #559: median must clear floor)."""
    med = {k: v.get("median", float("nan")) for k, v in summary.items()}
    return _verdict_for_metrics(med)


def walk_forward(
    daily_returns: pd.Series,
    spy_returns: pd.Series,
    fund_rate: pd.Series,
    train_months: int = 36,
    oot_months: int = 12,
    step_months: int = 6,
    margin_spread_bps: int = 150,
) -> WalkForwardResult:
    """Rolling walk-forward.

    daily_returns and spy_returns are pd.Series indexed by date (daily).
    fund_rate is a pd.Series indexed by date in PERCENT (e.g. DFF=5.33 means 5.33%).
    """
    # align
    df = pd.concat(
        [daily_returns.rename("strat"), spy_returns.rename("spy"), fund_rate.rename("fund")],
        axis=1, join="inner",
    ).dropna(subset=["strat", "spy"])
    df = df.sort_index()
    df["fund"] = df["fund"].ffill().bfill()
    if df.empty:
        return WalkForwardResult(per_fold=[], summary={}, verdict="FAILS CALMAR FLOOR")

    start = df.index.min()
    end = df.index.max()

    per_fold: List[Dict] = []
    cursor = start
    while True:
        tr_start = cursor
        tr_end = tr_start + pd.DateOffset(months=train_months)
        oot_start = tr_end
        oot_end = oot_start + pd.DateOffset(months=oot_months)
        if oot_end > end + pd.Timedelta(days=1):
            break
        oot_mask = (df.index >= oot_start) & (df.index < oot_end)
        oot = df.loc[oot_mask]
        if len(oot) < 20:
            cursor = cursor + pd.DateOffset(months=step_months)
            continue

        strat_m = _metrics(oot["strat"])
        spy1 = _spy_levered(oot["spy"], 1.0, oot["fund"], margin_spread_bps)
        spy15 = _spy_levered(oot["spy"], 1.5, oot["fund"], margin_spread_bps)
        spy2 = _spy_levered(oot["spy"], 2.0, oot["fund"], margin_spread_bps)
        spy1_m = _metrics(spy1)
        spy15_m = _metrics(spy15)
        spy2_m = _metrics(spy2)

        per_fold.append({
            "train_start": str(tr_start.date()),
            "train_end": str(tr_end.date()),
            "oot_start": str(oot_start.date()),
            "oot_end": str(oot_end.date()),
            **strat_m,
            "spy_1x_sharpe": spy1_m["sharpe"],
            "spy_1x_cagr": spy1_m["cagr"],
            "spy_15x_sharpe": spy15_m["sharpe"],
            "spy_15x_cagr": spy15_m["cagr"],
            "spy_2x_sharpe": spy2_m["sharpe"],
            "spy_2x_cagr": spy2_m["cagr"],
        })
        cursor = cursor + pd.DateOffset(months=step_months)

    summary = _summarize(per_fold)
    verdict = _summary_verdict(summary)
    return WalkForwardResult(per_fold=per_fold, summary=summary, verdict=verdict)


# ---------------------------------------------------------------------------
# sanity test: WF on SPY itself
# ---------------------------------------------------------------------------
def _sanity_test():
    ROOT = Path("/home/jupiter/Lvl3Quant")
    prices = pd.read_parquet(ROOT / "wheel_strategy_v1/data/cache/prices.parquet")
    spy = prices[prices["ticker"] == "SPY"].sort_values("date").set_index("date")["close"]
    if spy.empty:
        # fallback: build a synthetic SPY from average of all tickers' close
        agg = prices.groupby("date")["close"].mean()
        spy_ret = agg.pct_change().dropna()
    else:
        spy_ret = spy.pct_change().dropna()

    # try to get a FRED-ish fund rate; otherwise constant 5%
    me = pd.read_parquet(ROOT / "wheel_strategy_v1/data/cache/macro_extra.parquet").set_index("date")
    if "fed_funds" in me.columns:
        fund = me["fed_funds"].astype(float).reindex(spy_ret.index).ffill().bfill().fillna(5.0)
    else:
        fund = pd.Series(5.0, index=spy_ret.index)

    # WF on SPY-as-strategy — should produce VIABLE BUT UNDER-TARGET / FAILS CALMAR FLOOR
    res = walk_forward(
        daily_returns=spy_ret,
        spy_returns=spy_ret,
        fund_rate=fund,
        train_months=36,
        oot_months=12,
        step_months=6,
    )
    print(f"folds: {len(res.per_fold)}")
    print(f"verdict: {res.verdict}")
    print("summary medians:")
    for k in ["sharpe", "sortino", "cagr", "max_dd", "calmar", "pf", "wr"]:
        v = res.summary.get(k, {})
        print(f"  {k:>8s}: med={v.get('median', float('nan')):.3f}  p25={v.get('p25', float('nan')):.3f}  p75={v.get('p75', float('nan')):.3f}")


if __name__ == "__main__":
    _sanity_test()
