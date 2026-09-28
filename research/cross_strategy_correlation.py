#!/usr/bin/env python3
"""
Cross-Strategy Correlation Analysis
====================================
Analyzes diversification benefit across portfolio strategies.

Strategies:
  1. ETF Rotation v3 (sector rotation, monthly rebalance)
  2. BPS Dynamic (bull put spread, options income)
  3. V5 CSP Combined (cash-secured puts, multi-factor)
  4. CSP 30DTE (cash-secured puts, 30-day tenor)
  5. Wheel RV-Gated SPY (regime-gated wheel on SPY)
  6. SPY Buy-and-Hold (benchmark)

Author: Claude (cross-strategy research)
Date: 2026-07-09
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
import json

# ── Configuration ──────────────────────────────────────────────────────
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/research")
OUTPUT_DIR.mkdir(exist_ok=True)

ROLLING_WINDOW = 60  # trading days

# ── Load strategy equity curves ───────────────────────────────────────

def load_equity(path, date_col="date", equity_col="equity"):
    """Load equity curve and compute daily returns."""
    df = pd.read_parquet(path)
    if date_col not in df.columns and df.index.name == date_col:
        df = df.reset_index()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    df["ret"] = df[equity_col].pct_change()
    return df[["ret"]].dropna()


def load_etf_rotation():
    """ETF Rotation v3 — has daily_ret directly."""
    path = "/home/jupiter/Lvl3Quant/output/macro_picker/etf_rotation_quality_20260709_223944/book.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df = df.rename(columns={"daily_ret": "ret"})
    return df[["ret"]].dropna()


def load_spy_buyhold():
    """SPY buy-and-hold from yfinance (cached or download)."""
    cache_path = OUTPUT_DIR / "spy_daily_returns.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if len(df) > 1000:
            return df

    try:
        import yfinance as yf
        spy = yf.download("SPY", start="2016-01-01", end="2026-07-09", progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        spy = spy[["Close"]].dropna()
        spy["ret"] = spy["Close"].pct_change()
        spy = spy[["ret"]].dropna()
        spy.index.name = "date"
        spy.to_parquet(cache_path)
        return spy
    except Exception as e:
        print(f"  [WARN] yfinance failed ({e}), using ETF rotation SPY benchmark proxy")
        # Fallback: use the wheel RV gate's spy_ret column
        df = pd.read_parquet("/home/jupiter/Lvl3Quant/output/wheel_regime_gated_spy_v1/equity_RV_GATE.parquet")
        if "spy_ret" in df.columns:
            df = df[["spy_ret"]].rename(columns={"spy_ret": "ret"}).dropna()
            df.index.name = "date"
            return df
        raise


print("=" * 70)
print("CROSS-STRATEGY CORRELATION ANALYSIS")
print("=" * 70)

# Load all strategies
print("\nLoading strategy equity curves...")
strategies = {}

try:
    strategies["ETF_Rotation_v3"] = load_etf_rotation()
    print(f"  ETF Rotation v3:   {len(strategies['ETF_Rotation_v3']):,} days  ({strategies['ETF_Rotation_v3'].index[0].date()} → {strategies['ETF_Rotation_v3'].index[-1].date()})")
except Exception as e:
    print(f"  [SKIP] ETF Rotation v3: {e}")

try:
    strategies["BPS_Dynamic"] = load_equity("/home/jupiter/Lvl3Quant/output/bps_dynamic_delta/eq_E_dynamic_aggressive.parquet")
    print(f"  BPS Dynamic:       {len(strategies['BPS_Dynamic']):,} days  ({strategies['BPS_Dynamic'].index[0].date()} → {strategies['BPS_Dynamic'].index[-1].date()})")
except Exception as e:
    print(f"  [SKIP] BPS Dynamic: {e}")

try:
    strategies["V5_CSP_Combined"] = load_equity("/home/jupiter/Lvl3Quant/output/wheel_v5_research/equity_v5_combined.parquet")
    print(f"  V5 CSP Combined:   {len(strategies['V5_CSP_Combined']):,} days  ({strategies['V5_CSP_Combined'].index[0].date()} → {strategies['V5_CSP_Combined'].index[-1].date()})")
except Exception as e:
    print(f"  [SKIP] V5 CSP Combined: {e}")

try:
    strategies["CSP_30DTE"] = load_equity("/home/jupiter/Lvl3Quant/output/csp_vs_bps_real_costs/eq_CSP_30DTE_15pct.parquet")
    print(f"  CSP 30DTE:         {len(strategies['CSP_30DTE']):,} days  ({strategies['CSP_30DTE'].index[0].date()} → {strategies['CSP_30DTE'].index[-1].date()})")
except Exception as e:
    print(f"  [SKIP] CSP 30DTE: {e}")

try:
    strategies["Wheel_RV_Gate"] = load_equity("/home/jupiter/Lvl3Quant/output/wheel_regime_gated_spy_v1/equity_RV_GATE.parquet")
    print(f"  Wheel RV Gate:     {len(strategies['Wheel_RV_Gate']):,} days  ({strategies['Wheel_RV_Gate'].index[0].date()} → {strategies['Wheel_RV_Gate'].index[-1].date()})")
except Exception as e:
    print(f"  [SKIP] Wheel RV Gate: {e}")

try:
    strategies["SPY_BuyHold"] = load_spy_buyhold()
    print(f"  SPY Buy&Hold:      {len(strategies['SPY_BuyHold']):,} days  ({strategies['SPY_BuyHold'].index[0].date()} → {strategies['SPY_BuyHold'].index[-1].date()})")
except Exception as e:
    print(f"  [SKIP] SPY Buy&Hold: {e}")

if len(strategies) < 3:
    print("\n[ERROR] Need at least 3 strategies for meaningful correlation analysis.")
    exit(1)


# ── Build aligned returns matrix ──────────────────────────────────────

print(f"\nAligning {len(strategies)} strategies to common date range...")
returns = pd.DataFrame({name: s["ret"] for name, s in strategies.items()})
returns = returns.dropna(how="all")

# Find overlapping period
common = returns.dropna()
print(f"  Full union period: {returns.index[0].date()} → {returns.index[-1].date()} ({len(returns):,} days)")
print(f"  Common overlap:    {common.index[0].date()} → {common.index[-1].date()} ({len(common):,} days)")

# For correlation, use pairwise complete observations (not just common overlap)
# This preserves more data for pairs with longer histories

# ── 1. Pairwise Correlation Matrix ───────────────────────────────────

print("\n" + "=" * 70)
print("1. PAIRWISE CORRELATION MATRIX")
print("=" * 70)

corr_matrix = returns.corr(min_periods=60)
print(f"\n   (Using pairwise complete observations, min 60 days overlap)\n")

# Print formatted
names = list(corr_matrix.columns)
header = f"{'':>20s}" + "".join(f"{n:>16s}" for n in names)
print(header)
print("-" * len(header))
for row_name in names:
    row_str = f"{row_name:>20s}"
    for col_name in names:
        val = corr_matrix.loc[row_name, col_name]
        if row_name == col_name:
            row_str += f"{'1.000':>16s}"
        elif pd.isna(val):
            row_str += f"{'N/A':>16s}"
        else:
            row_str += f"{val:>16.3f}"
    print(row_str)

# ── 2. Individual Strategy Metrics ────────────────────────────────────

print("\n" + "=" * 70)
print("2. INDIVIDUAL STRATEGY METRICS (annualized)")
print("=" * 70)


def compute_metrics(daily_ret):
    """Compute annualized risk-adjusted metrics."""
    daily_ret = daily_ret.dropna()
    n = len(daily_ret)
    if n < 30:
        return {}

    ann_ret = daily_ret.mean() * 252
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = daily_ret[daily_ret < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    cum = (1 + daily_ret).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    wr = (daily_ret > 0).sum() / n

    # Profit factor
    gains = daily_ret[daily_ret > 0].sum()
    losses = abs(daily_ret[daily_ret < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    return {
        "CAGR": ann_ret,
        "Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": max_dd,
        "Calmar": calmar,
        "WR": wr,
        "PF": pf,
        "Days": n,
    }


metrics = {}
for name, s in strategies.items():
    metrics[name] = compute_metrics(s["ret"])

metrics_df = pd.DataFrame(metrics).T
fmt_cols = {
    "CAGR": "{:.1%}",
    "Vol": "{:.1%}",
    "Sharpe": "{:.2f}",
    "Sortino": "{:.2f}",
    "MaxDD": "{:.1%}",
    "Calmar": "{:.2f}",
    "WR": "{:.1%}",
    "PF": "{:.2f}",
    "Days": "{:.0f}",
}

print(f"\n{'Strategy':>20s}" + "".join(f"{c:>10s}" for c in fmt_cols))
print("-" * (20 + 10 * len(fmt_cols)))
for name, row in metrics_df.iterrows():
    line = f"{name:>20s}"
    for col, fmt in fmt_cols.items():
        val = row.get(col, np.nan)
        if pd.isna(val):
            line += f"{'N/A':>10s}"
        else:
            line += f"{fmt.format(val):>10s}"
    print(line)


# ── 3. Rolling Correlation (60d) ─────────────────────────────────────

print("\n" + "=" * 70)
print("3. ROLLING CORRELATION ANALYSIS (60d window)")
print("=" * 70)

# Compute rolling correlation of each strategy vs SPY
spy_col = "SPY_BuyHold"
if spy_col in returns.columns:
    print(f"\n   Rolling correlation vs SPY (60d window):\n")
    print(f"{'Strategy':>20s}  {'Mean':>8s}  {'Median':>8s}  {'Min':>8s}  {'Max':>8s}  {'Std':>8s}  {'% > 0.5':>8s}")
    print("-" * 78)

    rolling_corrs = {}
    for name in returns.columns:
        if name == spy_col:
            continue
        rc = returns[name].rolling(ROLLING_WINDOW).corr(returns[spy_col]).dropna()
        if len(rc) > 30:
            rolling_corrs[name] = rc
            pct_high = (rc > 0.5).mean()
            print(f"{name:>20s}  {rc.mean():>8.3f}  {rc.median():>8.3f}  {rc.min():>8.3f}  {rc.max():>8.3f}  {rc.std():>8.3f}  {pct_high:>7.1%}")

    # Cross-strategy rolling correlations (non-SPY pairs)
    non_spy = [c for c in returns.columns if c != spy_col]
    if len(non_spy) >= 2:
        print(f"\n   Rolling correlation between strategy pairs (60d window):\n")
        print(f"{'Pair':>40s}  {'Mean':>8s}  {'Median':>8s}  {'Min':>8s}  {'Max':>8s}")
        print("-" * 72)
        for i, a in enumerate(non_spy):
            for b in non_spy[i + 1:]:
                rc = returns[a].rolling(ROLLING_WINDOW).corr(returns[b]).dropna()
                if len(rc) > 30:
                    pair_name = f"{a[:18]} / {b[:18]}"
                    print(f"{pair_name:>40s}  {rc.mean():>8.3f}  {rc.median():>8.3f}  {rc.min():>8.3f}  {rc.max():>8.3f}")


# ── 4. Combined Portfolio Analysis ────────────────────────────────────

print("\n" + "=" * 70)
print("4. COMBINED PORTFOLIO (EQUAL-WEIGHT)")
print("=" * 70)

# Exclude SPY benchmark from portfolio
portfolio_strats = [c for c in returns.columns if c != spy_col]
port_returns = returns[portfolio_strats].dropna()

if len(port_returns) > 60:
    # Equal-weight portfolio
    port_returns["EqualWeight"] = port_returns[portfolio_strats].mean(axis=1)

    port_metrics = compute_metrics(port_returns["EqualWeight"])
    print(f"\n   Equal-weight portfolio of {len(portfolio_strats)} strategies:")
    print(f"   Period: {port_returns.index[0].date()} → {port_returns.index[-1].date()} ({len(port_returns):,} days)\n")

    for k, fmt in fmt_cols.items():
        val = port_metrics.get(k, np.nan)
        if not pd.isna(val):
            print(f"   {k:>10s}: {fmt.format(val)}")

    # Compare to individual
    individual_sharpes = [metrics[s]["Sharpe"] for s in portfolio_strats if s in metrics and "Sharpe" in metrics[s]]
    avg_ind_sharpe = np.mean(individual_sharpes) if individual_sharpes else 0
    port_sharpe = port_metrics.get("Sharpe", 0)

    print(f"\n   {'Metric':>30s}  {'Value':>10s}")
    print(f"   {'-' * 42}")
    print(f"   {'Portfolio Sharpe':>30s}  {port_sharpe:>10.2f}")
    print(f"   {'Avg Individual Sharpe':>30s}  {avg_ind_sharpe:>10.2f}")

    div_ratio = port_sharpe / avg_ind_sharpe if avg_ind_sharpe > 0 else float("inf")
    print(f"   {'Diversification Ratio':>30s}  {div_ratio:>10.2f}")
    print(f"   {'(>1 = diversification benefit)':>30s}")

    # Best individual vs portfolio
    best_ind_name = max(portfolio_strats, key=lambda s: metrics.get(s, {}).get("Sharpe", -999))
    best_ind_sharpe = metrics.get(best_ind_name, {}).get("Sharpe", 0)
    print(f"\n   {'Best Individual':>30s}  {best_ind_name} (Sharpe {best_ind_sharpe:.2f})")
    print(f"   {'Portfolio beats best?':>30s}  {'YES' if port_sharpe > best_ind_sharpe else 'NO'} (Portfolio: {port_sharpe:.2f})")


# ── 5. Crisis Correlation Analysis ────────────────────────────────────

print("\n" + "=" * 70)
print("5. CRISIS CORRELATION (DRAWDOWN REGIME ANALYSIS)")
print("=" * 70)

if spy_col in returns.columns:
    spy_ret = returns[spy_col].dropna()

    # Define crisis: SPY drawdown > 5% from recent high (20d)
    spy_cum = (1 + spy_ret).cumprod()
    spy_high = spy_cum.rolling(60, min_periods=1).max()
    spy_dd = (spy_cum - spy_high) / spy_high

    crisis_mask = spy_dd < -0.05
    normal_mask = spy_dd >= -0.02

    crisis_dates = crisis_mask[crisis_mask].index
    normal_dates = normal_mask[normal_mask].index

    print(f"\n   Crisis defined as: SPY drawdown > 5% from 60d high")
    print(f"   Crisis days: {len(crisis_dates):,}")
    print(f"   Normal days: {len(normal_dates):,}")

    if len(crisis_dates) > 30 and len(normal_dates) > 30:
        print(f"\n   {'Strategy':>20s}  {'Corr(normal)':>14s}  {'Corr(crisis)':>14s}  {'Delta':>8s}  {'Verdict':>12s}")
        print("   " + "-" * 72)

        for name in portfolio_strats:
            # Get aligned data
            pair = returns[[name, spy_col]].dropna()
            pair_crisis = pair.loc[pair.index.isin(crisis_dates)]
            pair_normal = pair.loc[pair.index.isin(normal_dates)]

            if len(pair_crisis) > 20 and len(pair_normal) > 20:
                corr_crisis = pair_crisis[name].corr(pair_crisis[spy_col])
                corr_normal = pair_normal[name].corr(pair_normal[spy_col])
                delta = corr_crisis - corr_normal

                if delta > 0.15:
                    verdict = "BAD (↑ crisis)"
                elif delta < -0.10:
                    verdict = "GOOD (↓ crisis)"
                else:
                    verdict = "Stable"

                print(f"   {name:>20s}  {corr_normal:>14.3f}  {corr_crisis:>14.3f}  {delta:>+8.3f}  {verdict:>12s}")

    # Drawdown contribution analysis
    print(f"\n   Portfolio drawdown vs SPY drawdown (worst 20 SPY days):")
    spy_worst = spy_ret.nsmallest(20)
    print(f"\n   {'Date':>12s}  {'SPY':>8s}", end="")
    for s in portfolio_strats:
        print(f"  {s[:12]:>12s}", end="")
    print(f"  {'Portfolio':>10s}")
    print("   " + "-" * (22 + 14 * len(portfolio_strats) + 10))

    for date, spy_val in spy_worst.items():
        line = f"   {date.strftime('%Y-%m-%d'):>12s}  {spy_val:>+8.2%}"
        port_day = []
        for s in portfolio_strats:
            val = returns.loc[returns.index == date, s]
            if len(val) > 0 and not pd.isna(val.iloc[0]):
                line += f"  {val.iloc[0]:>+12.2%}"
                port_day.append(val.iloc[0])
            else:
                line += f"  {'N/A':>12s}"
        if port_day:
            line += f"  {np.mean(port_day):>+10.2%}"
        print(line)


# ── 6. Summary & Recommendations ─────────────────────────────────────

print("\n" + "=" * 70)
print("6. SUMMARY & KEY FINDINGS")
print("=" * 70)

# Find truly independent strategies (corr < 0.3)
print("\n   TRULY INDEPENDENT PAIRS (|correlation| < 0.30):")
independent_found = False
for i, a in enumerate(corr_matrix.columns):
    for b in list(corr_matrix.columns)[i + 1:]:
        val = corr_matrix.loc[a, b]
        if not pd.isna(val) and abs(val) < 0.30:
            print(f"     {a} ↔ {b}: {val:.3f}")
            independent_found = True
if not independent_found:
    print("     None found — all strategies are moderately+ correlated")

# Find highly correlated (redundant) pairs
print("\n   REDUNDANT PAIRS (|correlation| > 0.70):")
redundant_found = False
for i, a in enumerate(corr_matrix.columns):
    for b in list(corr_matrix.columns)[i + 1:]:
        val = corr_matrix.loc[a, b]
        if not pd.isna(val) and abs(val) > 0.70:
            print(f"     {a} ↔ {b}: {val:.3f}  ← consider dropping one")
            redundant_found = True
if not redundant_found:
    print("     None found — all strategies have distinct behavior")

# Average pairwise correlation (excluding diagonal and SPY)
non_spy_strats = [c for c in corr_matrix.columns if c != spy_col]
pair_corrs = []
for i, a in enumerate(non_spy_strats):
    for b in non_spy_strats[i + 1:]:
        val = corr_matrix.loc[a, b]
        if not pd.isna(val):
            pair_corrs.append(val)

if pair_corrs:
    avg_pair_corr = np.mean(pair_corrs)
    print(f"\n   Average pairwise correlation (strategies only): {avg_pair_corr:.3f}")
    if avg_pair_corr < 0.3:
        print("   → EXCELLENT diversification — strategies are largely independent")
    elif avg_pair_corr < 0.5:
        print("   → GOOD diversification — moderate independence between strategies")
    elif avg_pair_corr < 0.7:
        print("   → FAIR diversification — some redundancy, limited benefit from combining")
    else:
        print("   → POOR diversification — strategies are highly correlated (same risk)")

print("\n" + "=" * 70)
print("Analysis complete.")
print("=" * 70)


# ── Save results to JSON ─────────────────────────────────────────────

results = {
    "analysis_date": "2026-07-09",
    "strategies_analyzed": list(strategies.keys()),
    "common_period": {
        "start": str(common.index[0].date()) if len(common) > 0 else None,
        "end": str(common.index[-1].date()) if len(common) > 0 else None,
        "n_days": len(common),
    },
    "correlation_matrix": corr_matrix.to_dict(),
    "individual_metrics": {k: {mk: float(mv) if not pd.isna(mv) else None for mk, mv in v.items()} for k, v in metrics.items()},
    "portfolio_metrics": {k: float(v) if not pd.isna(v) else None for k, v in port_metrics.items()} if 'port_metrics' in dir() else {},
    "diversification_ratio": float(div_ratio) if 'div_ratio' in dir() else None,
    "avg_pairwise_correlation": float(avg_pair_corr) if pair_corrs else None,
}

results_path = OUTPUT_DIR / "cross_strategy_correlation_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {results_path}")
