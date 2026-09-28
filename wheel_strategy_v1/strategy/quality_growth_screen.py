"""
quality_growth_screen.py — HC #557 R2 attempt to tighten the regime-symmetry gate.

Hypothesis: the v7 Balanced tier fails HC #557 R2 (regime-symmetry sub-gate) because the
broad 70-name universe contains weak balance-sheet / unprofitable names whose assignments
during red months drag the red-bucket Sharpe far below the green-bucket Sharpe.

Restricting the universe to quality-growth names — clean balance sheets, real profitability,
positive ebitda margins — should reduce assignment damage in red months and lift the worst-
bucket Sharpe closer to the best-bucket Sharpe.

Filter (applied on top of filter_balanced's fund_score >= 45 floor):
  - fund_score          >= 60.0           (top half of remaining universe)
  - ebitda_margin       > 0.10            (real operating profitability)
  - debt_to_equity      < 1.0             (clean balance sheet)
  - net_margin          > 0.0             (actually profitable)
  - market_cap          > 50e9            (large-cap, liquid)

We then re-run the Balanced FullWheel tier on the filtered universe, real IV + skew +
slippage + regime overlay, 2020-2025, $100k, and emit the same HC #557 SPY benchmark
comparison.

Usage:
    cd /home/jupiter/Lvl3Quant/wheel_strategy_v1
    python3 -m strategy.quality_growth_screen
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List
from dataclasses import replace

import numpy as np
import pandas as pd

from strategy.tiers import balanced_tier, _full_wheel, TierSpec, _has_iv_history
from strategy.tier_runner import (
    _load_inputs, _apply_iv_rank_floor, compute_metrics,
)
from backtest.wheel_engine import run_wheel

ROOT = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1")
RESULTS_DIR = ROOT / "results" / "quality_growth_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- quality-growth filter ----------

def filter_quality_growth(universe: pd.DataFrame,
                          fundamentals: pd.DataFrame,
                          iv_features: pd.DataFrame,
                          prices: pd.DataFrame) -> List[str]:
    """Restrict balanced-tier universe to quality-growth names."""
    fs = fundamentals.set_index("ticker")
    keep = []
    rejected = {"no_fund": 0, "no_iv": 0, "fund_score": 0, "ebitda_margin": 0,
                "debt_to_equity": 0, "net_margin": 0, "market_cap": 0}
    for tk in universe["ticker"]:
        if tk not in fs.index:
            rejected["no_fund"] += 1
            continue
        if not _has_iv_history(iv_features, tk):
            rejected["no_iv"] += 1
            continue
        row = fs.loc[tk]
        fund_score = float(row.get("fund_score", np.nan))
        ebitda_m = float(row.get("ebitda_margin", np.nan))
        de = float(row.get("debt_to_equity", np.nan))
        net_m = float(row.get("net_margin", np.nan))
        mc = float(row.get("market_cap", np.nan))
        if not (fund_score >= 60.0):
            rejected["fund_score"] += 1; continue
        if not (ebitda_m > 0.10):
            rejected["ebitda_margin"] += 1; continue
        if not (de < 1.0):
            rejected["debt_to_equity"] += 1; continue
        if not (net_m > 0.0):
            rejected["net_margin"] += 1; continue
        if not (mc > 50e9):
            rejected["market_cap"] += 1; continue
        keep.append(tk)
    print(f"[QG screen] kept {len(keep)} / {len(universe)} tickers")
    print(f"[QG screen] rejections: {rejected}")
    return sorted(keep)


def quality_growth_balanced_tier() -> TierSpec:
    base = balanced_tier()
    qg = TierSpec(
        name="Tier2_Balanced_QG",
        target_annual_yield_pct=base.target_annual_yield_pct,
        description=f"{base.description} + quality-growth screen (fund>=60, ebitda>10%, D/E<1, net>0, mc>$50B)",
        wheel_cfg=base.wheel_cfg,
        universe_filter=filter_quality_growth,
        iv_rank_floor=base.iv_rank_floor,
        regime_sensitivity=base.regime_sensitivity,
        capital_share_default=base.capital_share_default,
    )
    return _full_wheel(qg)  # assignment-allowed mode, same as v7 canonical


# ---------- regime-bucket helpers (copied from hc557_spy_benchmark for self-contained run) ----------

def regime_classify(spy_close: pd.Series, thresh_sigma: float = 0.5) -> pd.Series:
    """Classify each trading day as green/red/flat by close-to-close return."""
    ret = spy_close.pct_change().dropna()
    sigma = ret.std()
    cutoff = thresh_sigma * sigma
    out = pd.Series("flat", index=ret.index, dtype=object)
    out[ret > cutoff] = "green"
    out[ret < -cutoff] = "red"
    return out


def monthly_income_by_regime(ledger_df: pd.DataFrame, regime: pd.Series) -> dict:
    """Aggregate realized-cash by month within each regime bucket."""
    if ledger_df is None or len(ledger_df) == 0:
        return {"green": np.nan, "red": np.nan, "flat": np.nan}
    df = ledger_df.copy()
    df["close_date"] = pd.to_datetime(df["close_date"])
    df = df.set_index("close_date")
    df["regime"] = regime.reindex(df.index, method="ffill")
    out = {}
    for bucket in ("green", "red", "flat"):
        sub = df[df["regime"] == bucket]
        if len(sub) == 0:
            out[bucket] = 0.0
            continue
        # Monthly sum of realized pnl, then mean across months in bucket
        monthly = sub["realized_pnl"].resample("M").sum()
        out[bucket] = float(monthly.mean()) if len(monthly) else 0.0
    return out


def stratified_sharpe(daily_ret: pd.Series, regime: pd.Series) -> dict:
    out = {}
    ann = np.sqrt(252)
    for bucket in ("green", "red", "flat"):
        days = regime[regime == bucket].index
        sub = daily_ret.reindex(days).dropna()
        if len(sub) < 5 or sub.std() == 0:
            out[bucket] = np.nan
            continue
        out[bucket] = float(sub.mean() / sub.std() * ann)
    return out


def verdict_for_tier(metrics: dict, spy_sharpe_15x: float, spy_mdd_15x: float,
                     monthly_income: dict, strat_sharpe: dict) -> str:
    """HC #557 R1 + R2 evaluation."""
    reasons = []
    # R1
    r1_sharpe = metrics["sharpe"] > spy_sharpe_15x
    r1_dd = metrics["max_dd"] > spy_mdd_15x  # less negative = better
    r1_pass = r1_sharpe and r1_dd
    if not r1_pass:
        if not r1_sharpe:
            reasons.append(f"Sharpe {metrics['sharpe']:.2f} <= margin-SPY 1.5x {spy_sharpe_15x:.2f}")
        if not r1_dd:
            reasons.append(f"MaxDD {metrics['max_dd']:.1%} worse than margin-SPY 1.5x {spy_mdd_15x:.1%}")
    # R2 income-sign
    if min(monthly_income.values()) <= 0:
        reasons.append(f"income negative in some bucket: {monthly_income}")
    # R2 sharpe-symmetry
    vals = [v for v in strat_sharpe.values() if not np.isnan(v)]
    if len(vals) >= 2:
        worst, best = min(vals, key=abs), max(vals, key=abs)
        if abs(best) > 0:
            ratio = abs(worst) / abs(best)
            if ratio < 0.50:
                reasons.append(f"regime sharpe asymmetric worst/best={ratio:.2f} < 0.50")
    if not reasons:
        return "WHEEL WINS"
    return "WHEEL LOSES — JUST USE MARGIN SPY (reasons: " + "; ".join(reasons) + ")"


# ---------- runner ----------

def main():
    # Match v7 canonical setup: real IV + skew + slippage + regime overlay
    import strategy.tier_runner as TR
    from backtest import wheel_engine as WE

    # Engine flags (v7 canonical)
    WE.USE_SKEW = True
    WE.USE_SLIPPAGE = True
    WE.SLIPPAGE_FRAC = 0.025
    WE.SLIPPAGE_MIN_TICKS = 0.03

    data = _load_inputs(modeled=True, smoke=False, real_iv=True)

    spec = quality_growth_balanced_tier()
    tickers = spec.universe_filter(data["universe"], data["fundamentals"],
                                   data["iv"], data["prices"])
    if not tickers:
        print("[QG] empty universe — aborting")
        return
    print(f"[QG] universe: {tickers}")

    px = data["prices"][data["prices"]["ticker"].isin(tickers)].copy()
    iv = data["iv"][data["iv"]["ticker"].isin(tickers)].copy()
    iv = _apply_iv_rank_floor(iv, spec.iv_rank_floor)

    print(f"[QG] running Balanced_QG: {len(tickers)} tickers, {len(px):,} price rows")
    result = run_wheel(
        cfg=spec.wheel_cfg,
        prices=px, iv=iv, macro=data["macro"],
        fundamentals=data["fundamentals"], universe=data["universe"],
        starting_cash=100_000.0,
        start="2020-01-01", end="2025-12-31",
        verbose=False,
    )
    metrics = compute_metrics(result, 100_000.0)
    print(f"[QG] metrics: {metrics}")

    # Persist core artifacts
    out_dir = RESULTS_DIR
    if "ledger" in result and result["ledger"] is not None:
        result["ledger"].to_parquet(out_dir / "ledger_balanced_qg.parquet")
    if "equity" in result and result["equity"] is not None:
        result["equity"].to_parquet(out_dir / "equity_balanced_qg.parquet")
    with open(out_dir / "metrics_balanced_qg.json", "w") as f:
        json.dump({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                   for k, v in metrics.items()}, f, indent=2, default=str)

    # Pull SPY-margin baseline from v7 hc557 results for direct comparison
    v7_results = ROOT / "results" / "tier_ladder_v7_REAL_SKEW_SLIP_REGIME" / "hc557_results.parquet"
    if v7_results.exists():
        hc = pd.read_parquet(v7_results)
        try:
            spy15 = hc[hc["strategy"].str.contains("1.5x", na=False)].iloc[0]
            spy_sharpe_15x = float(spy15["sharpe"])
            spy_mdd_15x = float(spy15["max_dd"])
            print(f"[QG] margin-SPY 1.5x: Sharpe={spy_sharpe_15x:.2f} MaxDD={spy_mdd_15x:.1%}")
        except Exception as e:
            print(f"[QG] couldn't parse v7 spy benchmark ({e}) — defaulting to 0.70 / -0.473")
            spy_sharpe_15x, spy_mdd_15x = 0.70, -0.4734
    else:
        spy_sharpe_15x, spy_mdd_15x = 0.70, -0.4734

    # Regime classification (SPY close-to-close)
    spy_px = data["prices"][data["prices"]["ticker"] == "SPY"].copy()
    spy_px["date"] = pd.to_datetime(spy_px["date"])
    spy_px = spy_px.sort_values("date").set_index("date")
    regime = regime_classify(spy_px["close"])

    daily_ret = None
    if "equity" in result and result["equity"] is not None:
        eq = result["equity"].copy()
        eq.index = pd.to_datetime(eq.index)
        daily_ret = eq["equity"].pct_change().dropna()

    monthly_income = monthly_income_by_regime(result.get("ledger"), regime)
    strat_sharpe = stratified_sharpe(daily_ret, regime) if daily_ret is not None else {"green": np.nan, "red": np.nan, "flat": np.nan}

    verdict = verdict_for_tier(metrics, spy_sharpe_15x, spy_mdd_15x, monthly_income, strat_sharpe)

    report = f"""# Quality-Growth Screen — HC #557 R2 Test

**Verdict: {verdict}**

Window: 2020-01-01 to 2025-12-31. Real IV + skew + slippage + regime overlay (v7 canonical). $100k.

## Universe
{len(tickers)} quality-growth names: {', '.join(tickers)}

Filter: fund_score >= 60, ebitda_margin > 10%, debt_to_equity < 1.0, net_margin > 0, market_cap > $50B.

## Headline

| Metric | Balanced (broad) | Balanced_QG (quality-growth) | SPY 1.5x margin |
|---|---|---|---|
| CAGR | 10.5% | {metrics['cagr']:.1%} | 18.5% |
| Sharpe | 1.02 | {metrics['sharpe']:.2f} | {spy_sharpe_15x:.2f} |
| Sortino | 0.41 | {metrics.get('sortino', float('nan')):.2f} | 0.87 |
| MaxDD | -11.9% | {metrics['max_dd']:.1%} | {spy_mdd_15x:.1%} |

## Monthly Realized Income by Regime
green: ${monthly_income['green']:,.0f} | red: ${monthly_income['red']:,.0f} | flat: ${monthly_income['flat']:,.0f}

## Stratified Sharpe by Regime
green: {strat_sharpe['green']:.2f} | red: {strat_sharpe['red']:.2f} | flat: {strat_sharpe['flat']:.2f}
"""
    (out_dir / "quality_growth_report.md").write_text(report)
    with open(out_dir / "quality_growth_results.json", "w") as f:
        json.dump({
            "verdict": verdict,
            "metrics": {k: (float(v) if isinstance(v, (np.floating, np.integer, float, int)) else str(v))
                        for k, v in metrics.items()},
            "monthly_income": monthly_income,
            "stratified_sharpe": strat_sharpe,
            "universe": tickers,
            "spy_15x_sharpe": spy_sharpe_15x,
            "spy_15x_mdd": spy_mdd_15x,
        }, f, indent=2, default=str)

    print(f"\n[QG] Report: {out_dir / 'quality_growth_report.md'}")
    print(f"[QG] {verdict}")


if __name__ == "__main__":
    main()
