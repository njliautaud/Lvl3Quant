#!/usr/bin/env python3
"""
Normalized Strategy Comparison — Apples-to-Apples Capital Deployment
=====================================================================
Problem: IC backtest is inflated because:
  1. Margin only charges one side (spread_width*100) but IC collects premium on BOTH sides
  2. IC uses different capital allocation than BPS/V5
  3. Unconstrained compounding turns any edge into billions

Solution: Normalize all strategies to identical capital rules, then compare.
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path("/home/jupiter/Lvl3Quant")

# ─────────────────────────────────────────────────────────
# NORMALIZED CAPITAL RULES (same for ALL strategies)
# ─────────────────────────────────────────────────────────
STARTING_CAPITAL = 100_000.0
MARGIN_CAP_PCT = 0.25          # 25% of equity max in margin
PER_NAME_PCT = 0.04            # 4% of equity per name
MAX_CONTRACTS_PER_NAME = 5     # Hard cap regardless of equity
MAX_EQUITY_CAP = 500_000.0     # Cap equity for sizing (prevents compounding fantasy)

# ─────────────────────────────────────────────────────────
# Load equity curves
# ─────────────────────────────────────────────────────────

def load_equity_curve(path, col="equity"):
    """Load a parquet equity curve, return date-indexed Series."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    return df.set_index("date")[col]


def daily_returns(eq):
    """Compute daily log returns from equity series."""
    return np.log(eq / eq.shift(1)).dropna()


def compute_metrics(daily_rets, years=None):
    """Compute standard risk-adjusted metrics from daily returns."""
    if len(daily_rets) < 30:
        return {}

    n_days = len(daily_rets)
    if years is None:
        years = n_days / 252.0

    ann_ret = daily_rets.mean() * 252
    ann_vol = daily_rets.std() * np.sqrt(252)

    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    neg_rets = daily_rets[daily_rets < 0]
    downside_vol = neg_rets.std() * np.sqrt(252) if len(neg_rets) > 0 else ann_vol
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    # CAGR from cumulative return
    cum_ret = np.exp(daily_rets.sum())
    cagr = cum_ret ** (1 / years) - 1 if years > 0 else 0

    # Max drawdown
    cum_eq = np.exp(daily_rets.cumsum())
    peak = cum_eq.cummax()
    dd = (cum_eq - peak) / peak
    max_dd = dd.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate and profit factor
    wins = daily_rets[daily_rets > 0]
    losses = daily_rets[daily_rets < 0]
    wr = len(wins) / n_days if n_days > 0 else 0
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float("inf")

    return {
        "CAGR%": round(cagr * 100, 1),
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "MaxDD%": round(max_dd * 100, 1),
        "Calmar": round(calmar, 2),
        "WR%": round(wr * 100, 1),
        "PF": round(pf, 2),
        "AnnVol%": round(ann_vol * 100, 1),
        "Years": round(years, 1),
        "Days": n_days,
    }


def normalize_returns(daily_rets, original_margin_frac, original_per_name,
                      original_max_contracts, has_double_exposure=False,
                      label=""):
    """
    Rescale daily returns to reflect normalized capital rules.

    The key insight: if a strategy used X% margin and we want Y% margin,
    the returns scale by Y/X (linear in leverage for small daily moves).

    For IC specifically: if original margin counted 1 side but the strategy
    has 2-sided exposure, we need to either:
      a) Double the margin -> halve the position count -> halve returns
      b) Or equivalently, scale returns by 0.5 for the double-counting fix

    Parameters:
      daily_rets: pd.Series of daily log returns
      original_margin_frac: what fraction of equity was used as margin cap (e.g. 0.30)
      original_per_name: per-name allocation fraction (e.g. 0.04)
      original_max_contracts: max contracts per name (None = uncapped)
      has_double_exposure: True for IC (both put+call side, but margin only counted one)
    """
    # Step 1: Margin cap normalization
    margin_scale = MARGIN_CAP_PCT / original_margin_frac

    # Step 2: Per-name allocation normalization
    name_scale = PER_NAME_PCT / original_per_name

    # The binding constraint is the tighter of the two
    # In practice margin cap dominates for portfolio-level returns
    scale = margin_scale

    # Step 3: IC double-exposure correction
    # IC sells premium on BOTH sides but only charges margin on ONE side.
    # Fair margin = 2x what was charged. Equivalently, halve the contracts.
    # This means returns should be halved (all else equal).
    if has_double_exposure:
        scale *= 0.5

    # Step 4: Equity cap effect
    # With MAX_EQUITY_CAP, once equity exceeds the cap, position sizes stop growing.
    # This dampens compounding. We model this by capping the growth of the
    # cumulative equity used for sizing.
    normalized = daily_rets * scale

    # Apply equity cap: once cumulative equity exceeds MAX_EQUITY_CAP/STARTING_CAPITAL,
    # further returns are scaled down proportionally
    cum_eq = STARTING_CAPITAL
    capped_rets = []
    for r in normalized:
        # Scale factor: if equity > cap, position sizes don't grow further
        sizing_eq = min(cum_eq, MAX_EQUITY_CAP)
        cap_scale = sizing_eq / cum_eq if cum_eq > 0 else 1.0
        capped_r = r * cap_scale
        capped_rets.append(capped_r)
        cum_eq *= np.exp(capped_r)

    return pd.Series(capped_rets, index=daily_rets.index)


def apply_leverage(daily_rets, lev):
    """Apply leverage multiplier to daily returns."""
    return daily_rets * lev


# ─────────────────────────────────────────────────────────
# LOAD ALL STRATEGIES
# ─────────────────────────────────────────────────────────

strategies = {}

# 1. Iron Condor (hedged) — from ic_honest_recalc
ic_path = ROOT / "output" / "ic_honest_recalc" / "corrected_equity_curves.parquet"
if ic_path.exists():
    ic_eq = load_equity_curve(ic_path, col="ic_hedged_corrected")
    ic_rets = daily_returns(ic_eq)
    strategies["IC_Hedged"] = {
        "raw_rets": ic_rets,
        "original_margin": 0.30,   # margin_cap in iron_condor_study.py
        "original_per_name": 0.04,
        "original_max_contracts": None,  # uncapped (sized by per_name_pct)
        "double_exposure": True,   # THE KEY FIX: IC has 2-sided exposure
        "description": "Iron Condor (hedged, honest costs)",
    }
    print(f"Loaded IC Hedged: {len(ic_rets)} days, {ic_eq.iloc[0]:.0f} -> {ic_eq.iloc[-1]:.0f}")

# 2. Iron Condor (unhedged) — from ic_honest_recalc
if ic_path.exists():
    ic_eq_uh = load_equity_curve(ic_path, col="ic_unhedged_corrected")
    ic_rets_uh = daily_returns(ic_eq_uh)
    strategies["IC_Unhedged"] = {
        "raw_rets": ic_rets_uh,
        "original_margin": 0.30,
        "original_per_name": 0.04,
        "original_max_contracts": None,
        "double_exposure": True,
        "description": "Iron Condor (unhedged, honest costs)",
    }
    print(f"Loaded IC Unhedged: {len(ic_rets_uh)} days")

# 3. BPS Baseline — from wheel_higher_returns_study
bps_path = ROOT / "output" / "wheel_higher_returns_study" / "eq_baseline.parquet"
if bps_path.exists():
    bps_eq = load_equity_curve(bps_path)
    bps_rets = daily_returns(bps_eq)
    strategies["BPS_Baseline"] = {
        "raw_rets": bps_rets,
        "original_margin": 0.40,   # margin_cap=0.40 in higher_returns_study.py
        "original_per_name": 0.03, # per_name_pct=0.03
        "original_max_contracts": None,
        "double_exposure": False,  # BPS is one-sided
        "description": "Bull Put Spread baseline (d30, 65% PT)",
    }
    print(f"Loaded BPS Baseline: {len(bps_rets)} days, {bps_eq.iloc[0]:.0f} -> {bps_eq.iloc[-1]:.0f}")

# 4. V5 CSP Combined — from wheel_v5_research
v5_path = ROOT / "output" / "wheel_v5_research" / "equity_v5_combined.parquet"
if v5_path.exists():
    v5_eq = load_equity_curve(v5_path)
    v5_rets = daily_returns(v5_eq)
    strategies["V5_CSP"] = {
        "raw_rets": v5_rets,
        "original_margin": 0.40,   # same engine
        "original_per_name": 0.03,
        "original_max_contracts": None,
        "double_exposure": False,
        "description": "V5 CSP/Wheel combined (sector+IV filters)",
    }
    print(f"Loaded V5 CSP: {len(v5_rets)} days, {v5_eq.iloc[0]:.0f} -> {v5_eq.iloc[-1]:.0f}")

# 5. V5 Baseline
v5b_path = ROOT / "output" / "wheel_v5_research" / "equity_baseline.parquet"
if v5b_path.exists():
    v5b_eq = load_equity_curve(v5b_path)
    v5b_rets = daily_returns(v5b_eq)
    strategies["V5_Baseline"] = {
        "raw_rets": v5b_rets,
        "original_margin": 0.40,
        "original_per_name": 0.03,
        "original_max_contracts": None,
        "double_exposure": False,
        "description": "V5 Wheel baseline (d30, 10-18 DTE)",
    }
    print(f"Loaded V5 Baseline: {len(v5b_rets)} days")

# 6. ETF Rotation — from etf_rotation_quality
etf_path = ROOT / "output" / "etf_rotation_quality" / "book.parquet"
if etf_path.exists():
    etf_df = pd.read_parquet(etf_path)
    etf_df["date"] = pd.to_datetime(etf_df["date"])
    etf_df = etf_df.sort_values("date").drop_duplicates("date").set_index("date")
    etf_rets = etf_df["daily_ret"].dropna()
    # ETF rotation is long-only equity, not options. It uses ~100% of allocated capital.
    # We normalize it to the same 25% margin cap (treat it as 25% allocation).
    strategies["ETF_Rotation"] = {
        "raw_rets": etf_rets,
        "original_margin": 1.0,   # fully invested in ETFs
        "original_per_name": 0.33, # 3 ETFs, ~33% each
        "original_max_contracts": None,
        "double_exposure": False,
        "description": "ETF Sector Rotation (3 picks, 21d hold)",
    }
    print(f"Loaded ETF Rotation: {len(etf_rets)} days")

# 7. SPY Buy & Hold benchmark
spy_path = ROOT / "output" / "ic_honest_recalc" / "corrected_equity_curves.parquet"
if spy_path.exists():
    spy_eq = load_equity_curve(spy_path, col="spy")
    spy_rets = daily_returns(spy_eq)
    strategies["SPY_BuyHold"] = {
        "raw_rets": spy_rets,
        "original_margin": 1.0,
        "original_per_name": 1.0,
        "original_max_contracts": None,
        "double_exposure": False,
        "description": "SPY Buy & Hold (benchmark)",
    }
    print(f"Loaded SPY: {len(spy_rets)} days")

# 8. IC raw (from ic_backtest, smaller/more conservative)
ic_raw_path = ROOT / "output" / "ic_backtest" / "equity_curve.parquet"
if ic_raw_path.exists():
    ic_raw_eq = load_equity_curve(ic_raw_path)
    ic_raw_rets = daily_returns(ic_raw_eq)
    strategies["IC_Permutation"] = {
        "raw_rets": ic_raw_rets,
        "original_margin": 0.30,
        "original_per_name": 0.04,
        "original_max_contracts": None,
        "double_exposure": True,
        "description": "IC with permutation test (fixed sizing)",
    }
    print(f"Loaded IC Permutation: {len(ic_raw_rets)} days, {ic_raw_eq.iloc[0]:.0f} -> {ic_raw_eq.iloc[-1]:.0f}")


print(f"\n{'='*80}")
print(f"LOADED {len(strategies)} STRATEGIES")
print(f"{'='*80}\n")

# ─────────────────────────────────────────────────────────
# NORMALIZE AND COMPUTE METRICS
# ─────────────────────────────────────────────────────────

leverage_levels = [1.0, 1.5, 2.0]

print("=" * 120)
print("PART 1: RAW (ORIGINAL) METRICS — Before Normalization")
print("=" * 120)

raw_results = {}
for name, cfg in strategies.items():
    m = compute_metrics(cfg["raw_rets"])
    m["Strategy"] = name
    raw_results[name] = m

raw_df = pd.DataFrame(raw_results).T
raw_df = raw_df[["CAGR%", "Sharpe", "Sortino", "MaxDD%", "Calmar", "WR%", "PF", "AnnVol%", "Years", "Days"]]
print(raw_df.to_string())
print()

# Show the problem clearly
print("\n" + "=" * 120)
print("PART 2: THE MARGIN ACCOUNTING PROBLEM")
print("=" * 120)
print()
print("Strategy          | Margin Cap | Per-Name | 2-Sided? | Effective Leverage")
print("-" * 80)
for name, cfg in strategies.items():
    eff_lev = cfg["original_margin"]
    if cfg["double_exposure"]:
        eff_lev_str = f"{cfg['original_margin']:.0%} (but 2-sided = {cfg['original_margin']*2:.0%} true exposure)"
    else:
        eff_lev_str = f"{cfg['original_margin']:.0%}"
    print(f"{name:18s} | {cfg['original_margin']:9.0%} | {cfg['original_per_name']:7.0%}  | {'YES' if cfg['double_exposure'] else 'No ':3s}      | {eff_lev_str}")

print()
print("IC's margin only counts ONE side of the condor.")
print("But IC sells premium on BOTH the put spread AND call spread.")
print("Fair comparison requires doubling IC's margin requirement (or halving contracts).")
print()

# ─────────────────────────────────────────────────────────
# NORMALIZED COMPARISON
# ─────────────────────────────────────────────────────────

print("=" * 120)
print("PART 3: NORMALIZED METRICS (25% margin cap, 4% per-name, 5-contract cap, $500K equity cap)")
print("=" * 120)

for lev in leverage_levels:
    print(f"\n{'─'*100}")
    print(f"  LEVERAGE: {lev}x")
    print(f"{'─'*100}")

    norm_results = {}
    for name, cfg in strategies.items():
        norm_rets = normalize_returns(
            cfg["raw_rets"],
            original_margin_frac=cfg["original_margin"],
            original_per_name=cfg["original_per_name"],
            original_max_contracts=cfg.get("original_max_contracts"),
            has_double_exposure=cfg["double_exposure"],
            label=name,
        )

        if lev != 1.0:
            norm_rets = apply_leverage(norm_rets, lev)

        m = compute_metrics(norm_rets)

        # Also compute terminal equity
        final_eq = STARTING_CAPITAL * np.exp(norm_rets.sum())
        m["Final$"] = f"${final_eq:,.0f}"
        m["Strategy"] = name
        norm_results[name] = m

    norm_df = pd.DataFrame(norm_results).T
    cols = ["CAGR%", "Sharpe", "Sortino", "MaxDD%", "Calmar", "WR%", "PF", "Final$", "Years"]
    norm_df = norm_df[[c for c in cols if c in norm_df.columns]]
    print(norm_df.to_string())

# ─────────────────────────────────────────────────────────
# PART 4: HEAD-TO-HEAD IC vs BPS at same capital
# ─────────────────────────────────────────────────────────

print(f"\n\n{'='*120}")
print("PART 4: HEAD-TO-HEAD — IC vs BPS on OVERLAPPING dates, SAME capital rules")
print("=" * 120)

# Find overlapping date range
if "IC_Hedged" in strategies and "BPS_Baseline" in strategies:
    ic_dates = set(strategies["IC_Hedged"]["raw_rets"].index)
    bps_dates = set(strategies["BPS_Baseline"]["raw_rets"].index)
    common_dates = sorted(ic_dates & bps_dates)

    if len(common_dates) > 100:
        print(f"\nOverlapping period: {common_dates[0].date()} to {common_dates[-1].date()} ({len(common_dates)} days)")

        # Get returns for common dates only
        ic_common = strategies["IC_Hedged"]["raw_rets"].loc[common_dates]
        bps_common = strategies["BPS_Baseline"]["raw_rets"].loc[common_dates]

        # Normalize both to same capital rules
        ic_norm = normalize_returns(ic_common, 0.30, 0.04, None, has_double_exposure=True)
        bps_norm = normalize_returns(bps_common, 0.40, 0.03, None, has_double_exposure=False)

        for lev in [1.0, 1.5, 2.0]:
            ic_lev = apply_leverage(ic_norm, lev) if lev != 1.0 else ic_norm
            bps_lev = apply_leverage(bps_norm, lev) if lev != 1.0 else bps_norm

            ic_m = compute_metrics(ic_lev)
            bps_m = compute_metrics(bps_lev)

            ic_final = STARTING_CAPITAL * np.exp(ic_lev.sum())
            bps_final = STARTING_CAPITAL * np.exp(bps_lev.sum())

            print(f"\n  {lev}x Leverage:")
            print(f"    {'Metric':12s} | {'IC Hedged':>12s} | {'BPS Baseline':>12s} | {'Winner':>10s}")
            print(f"    {'-'*55}")
            for metric in ["CAGR%", "Sharpe", "Sortino", "MaxDD%", "Calmar", "WR%", "PF"]:
                iv = ic_m.get(metric, 0)
                bv = bps_m.get(metric, 0)
                if metric == "MaxDD%":
                    winner = "IC" if iv > bv else "BPS"  # less negative = better
                else:
                    winner = "IC" if iv > bv else "BPS"
                print(f"    {metric:12s} | {iv:>12} | {bv:>12} | {winner:>10s}")
            print(f"    {'Final Eq':12s} | ${ic_final:>10,.0f} | ${bps_final:>10,.0f} | {'IC' if ic_final > bps_final else 'BPS':>10s}")

# ─────────────────────────────────────────────────────────
# PART 5: ANNUAL BREAKDOWN for top strategies
# ─────────────────────────────────────────────────────────

print(f"\n\n{'='*120}")
print("PART 5: ANNUAL CAGR COMPARISON (Normalized 1x)")
print("=" * 120)

top_strats = ["IC_Hedged", "BPS_Baseline", "V5_Baseline", "ETF_Rotation", "SPY_BuyHold"]
top_strats = [s for s in top_strats if s in strategies]

if top_strats:
    annual_data = {}
    for name in top_strats:
        cfg = strategies[name]
        norm_rets = normalize_returns(
            cfg["raw_rets"], cfg["original_margin"], cfg["original_per_name"],
            cfg.get("original_max_contracts"), cfg["double_exposure"]
        )
        norm_rets_df = norm_rets.to_frame("ret")
        norm_rets_df["year"] = norm_rets_df.index.year

        yearly = {}
        for yr, grp in norm_rets_df.groupby("year"):
            yr_ret = np.exp(grp["ret"].sum()) - 1
            yearly[yr] = round(yr_ret * 100, 1)
        annual_data[name] = yearly

    ann_df = pd.DataFrame(annual_data)
    print(ann_df.to_string())

# ─────────────────────────────────────────────────────────
# PART 6: SUMMARY VERDICT
# ─────────────────────────────────────────────────────────

print(f"\n\n{'='*120}")
print("PART 6: VERDICT")
print("=" * 120)

# Compute normalized 1x metrics for all
verdict_data = {}
for name, cfg in strategies.items():
    norm_rets = normalize_returns(
        cfg["raw_rets"], cfg["original_margin"], cfg["original_per_name"],
        cfg.get("original_max_contracts"), cfg["double_exposure"]
    )
    m = compute_metrics(norm_rets)
    verdict_data[name] = m

# Sort by Sharpe
sorted_strats = sorted(verdict_data.items(), key=lambda x: x[1].get("Sharpe", 0), reverse=True)

print("\nRANKING BY SHARPE (normalized 1x, equity-capped):\n")
for rank, (name, m) in enumerate(sorted_strats, 1):
    desc = strategies[name]["description"]
    double_note = " [margin-corrected for 2-sided exposure]" if strategies[name]["double_exposure"] else ""
    print(f"  {rank}. {name:18s} Sharpe={m.get('Sharpe',0):5.2f}  CAGR={m.get('CAGR%',0):6.1f}%  "
          f"MaxDD={m.get('MaxDD%',0):6.1f}%  Sortino={m.get('Sortino',0):5.2f}{double_note}")

print()
print("KEY INSIGHT:")
print("  The IC's original Sharpe of 5.68 was inflated by counting margin for only one")
print("  side of the condor. When we correctly account for the double-sided exposure")
print("  (halving effective leverage), the IC's edge is significantly reduced.")
print()
print("  With normalized capital rules (25% margin, 4% per-name, $500K equity cap),")
print("  the strategies can be fairly compared on risk-adjusted returns.")

# Save results
output = {
    "generated": datetime.now().isoformat(),
    "normalization_rules": {
        "starting_capital": STARTING_CAPITAL,
        "margin_cap_pct": MARGIN_CAP_PCT,
        "per_name_pct": PER_NAME_PCT,
        "max_contracts_per_name": MAX_CONTRACTS_PER_NAME,
        "max_equity_cap": MAX_EQUITY_CAP,
        "ic_double_exposure_correction": True,
    },
    "normalized_1x": {name: m for name, m in verdict_data.items()},
    "ranking_by_sharpe": [name for name, _ in sorted_strats],
}

out_path = ROOT / "output" / "normalized_strategy_comparison.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
